import os, sys, io
import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor
import triton_python_backend_utils as pb_utils

class TritonPythonModel:
    def initialize(self, args):
        """
        Called once when the model is loaded.
        """
        self.model_name = args["model_name"]
        self.model_version = args["model_version"]
        self.model_repository = args["model_repository"]
        self.model_dir = os.path.join(self.model_repository, self.model_version)

        # Ensure local code (this version folder) is importable
        if self.model_dir not in sys.path:
            sys.path.insert(0, self.model_dir)

        # Import helpers after sys.path is set
        from groundingdino.util.inference import load_model, predict, annotate

        # Store on self and use these in execute()
        self._load_model = load_model
        self._predict = predict
        self._annotate = annotate

        cfg_path = os.path.join(self.model_dir, "groundingdino", "config", "GroundingDINO_SwinT_OGC.py")
        weights_path = os.path.join(self.model_dir, "weights", "groundingdino_swint_ogc.pth")

        if not os.path.isfile(cfg_path):
            raise RuntimeError(f"Config not found at {cfg_path}")
        if not os.path.isfile(weights_path):
            raise RuntimeError(f"Weights not found at {weights_path}")

        # Load model once (will use GPU if available via GroundingDINO code)
        self.model = self._load_model(cfg_path, weights_path)
        # Defaults if thresholds not provided
        self.default_box_thr = 0.05
        self.default_text_thr = 0.05

    @staticmethod
    def _bytes_from_input(obj_array):
        """
        Triton TYPE_STRING inputs arrive as numpy object arrays with elements that are
        bytes/bytearray or numpy scalars. Return a real Python bytes object.
        """
        if obj_array is None:
            return None
        if obj_array.size == 0:
            return None
        x = obj_array[0]
        if isinstance(x, (bytes, bytearray)):
            return bytes(x)
        # Some backends wrap as 0-d np.ndarray(object) or np.bytes_
        try:
            return x.tobytes()
        except Exception:
            # Last resort: if it’s a numpy scalar/string, convert to Python object first
            # (np.str_ -> str -> encode)
            if isinstance(x, np.str_):
                return str(x).encode("utf-8", errors="ignore")
            return bytes(x.tolist()) if hasattr(x, "tolist") else str(x).encode("utf-8", errors="ignore")

    @staticmethod
    def _decode_prompt(obj_array):
        """
        Decode TYPE_STRING [1] -> Python str (utf-8).
        """
        if obj_array is None or obj_array.size == 0:
            return ""
        raw = obj_array[0]
        if isinstance(raw, (bytes, bytearray)):
            return raw.decode("utf-8", errors="ignore")
        try:
            return raw.tobytes().decode("utf-8", errors="ignore")
        except Exception:
            # np.str_ or other scalar-like
            return str(raw)

    @staticmethod
    def _to_bytes_array(list_or_seq):
        """
        Convert a sequence of strings/bytes to a numpy object array of bytes
        compatible with Triton TYPE_STRING outputs.
        """
        out = []
        for p in list_or_seq:
            if isinstance(p, (bytes, bytearray)):
                out.append(bytes(p))
            else:
                out.append(str(p).encode("utf-8", errors="ignore"))
        return np.array(out, dtype=object)

    def execute(self, requests):
        responses = []
        for req in requests:
            try:
                # ---- Read inputs ----
                img_bytes_in = pb_utils.get_input_tensor_by_name(req, "image_bytes")
                prompt_in = pb_utils.get_input_tensor_by_name(req, "text_prompt")
                if img_bytes_in is None or prompt_in is None:
                    raise ValueError("Missing required inputs: image_bytes and/or text_prompt")

                image_bytes = self._bytes_from_input(img_bytes_in.as_numpy())
                prompt = self._decode_prompt(prompt_in.as_numpy())

                # Optional thresholds
                box_thr_t = pb_utils.get_input_tensor_by_name(req, "box_threshold")
                txt_thr_t = pb_utils.get_input_tensor_by_name(req, "text_threshold")
                box_thr = float(box_thr_t.as_numpy()[0]) if box_thr_t is not None else self.default_box_thr
                txt_thr = float(txt_thr_t.as_numpy()[0]) if txt_thr_t is not None else self.default_text_thr

                # ---- Decode image bytes -> RGB ----
                if image_bytes is None or len(image_bytes) == 0:
                    raise ValueError("image_bytes is empty")

                npbuf = np.frombuffer(image_bytes, dtype=np.uint8)
                bgr = cv2.imdecode(npbuf, cv2.IMREAD_COLOR)
                if bgr is None:
                    raise ValueError("Failed to decode image bytes (cv2.imdecode returned None)")
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

                # ---- Convert to Torch Tensor (for GroundingDINO predict) ----
                pil_img = Image.fromarray(rgb)
                tensor_img = to_tensor(pil_img).unsqueeze(0)  # [1, 3, H, W], float32 in [0,1]

                # ---- Run GroundingDINO ----
                boxes_t, logits_t, phrases = self._predict(
                    model=self.model,
                    image=tensor_img,   # pass torch tensor
                    caption=prompt,
                    box_threshold=box_thr,
                    text_threshold=txt_thr
                )

                # ---- Annotate BEFORE converting tensors (utils expect native types) ----
                annotated_frame = self._annotate(
                    image_source=rgb,
                    boxes=boxes_t,
                    logits=logits_t,
                    phrases=phrases
                )

                # ---- PNG encode annotated image ----
                ok, enc = cv2.imencode(".png", cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR))
                if not ok:
                    raise ValueError("Failed to encode annotated image")

                # ---- Convert outputs to NumPy on host ----
                if hasattr(boxes_t, "detach"):
                    boxes_np = boxes_t.detach().cpu().numpy().astype(np.float32)
                else:
                    boxes_np = np.asarray(boxes_t, dtype=np.float32)

                if hasattr(logits_t, "detach"):
                    logits_np = logits_t.detach().cpu().numpy().astype(np.float32)
                else:
                    logits_np = np.asarray(logits_t, dtype=np.float32)

                # phrases: sequence of strings -> bytes object array (TYPE_STRING)
                phrases_np = self._to_bytes_array(phrases if isinstance(phrases, (list, tuple)) else [phrases])
                # If no detections, return empty arrays for boxes/logits/phrases (allowed by dims [-1, ...])
                if boxes_np.size == 0:
                    boxes_np = boxes_np.reshape(0, 4).astype(np.float32)
                if logits_np.size == 0:
                    logits_np = logits_np.reshape(0,).astype(np.float32)
                # phrases already empty when no detections

                # ---- Build Triton outputs ----
                boxes_out = pb_utils.Tensor("boxes", boxes_np)
                logits_out = pb_utils.Tensor("logits", logits_np)
                phrases_out = pb_utils.Tensor("phrases", phrases_np)
                annotated_out = pb_utils.Tensor(
                    "annotated_image",
                    np.array([enc.tobytes()], dtype=object)  # TYPE_STRING dims: [1]
                )

                responses.append(
                    pb_utils.InferenceResponse(
                        output_tensors=[boxes_out, logits_out, phrases_out, annotated_out]
                    )
                )

            except Exception as e:
                # Return a TritonError so client sees HTTP 500 with message
                responses.append(
                    pb_utils.InferenceResponse(
                        output_tensors=[],
                        error=pb_utils.TritonError(f"{type(e).__name__}: {e}")
                    )
                )

        return responses
