import json
import numpy as np
import MNN
import time
from .text import UnicodeProcessor, length_to_mask, chunk_text


class MNNInference:
    def __init__(self, model_path, input_names, output_names) -> None:
        config = dict()
        config["backend"] = 0
        config["precision"] = "low"
        config["memory"] = "low"

        rt = MNN.nn.create_runtime_manager((config,))

        self.model = MNN.nn.load_module_from_file(
            model_path,
            input_names,
            output_names,
            runtime_manager=rt,
        )
        self.input_names = input_names

    def run(self, output_names, input_dict):
        mnn_inputs = []
        # Enforce correct input ordering based on self.input_names
        for key in self.input_names:
            value = input_dict[key]
            if key == "text_ids":
                data_type = MNN.numpy.int32
                value = value.astype(np.int32)
            else:
                data_type = MNN.numpy.float32

            mnn_placeholder = MNN.expr.placeholder(value.shape, dtype=data_type)
            mnn_placeholder.write(value)
            mnn_inputs.append(mnn_placeholder)

        output = self.model.forward(mnn_inputs)
        if len(output) <= 0:
            return None
        if len(output) > 0 and output[0].dtype == MNN.numpy.float32:
            return [np.array(output[0].read())]
        output = MNN.expr.convert(output[0], MNN.expr.NCHW)
        output = output.read()
        return [np.array(output)]


class Style:
    def __init__(self, style_ttl_onnx: np.ndarray, style_dp_onnx: np.ndarray):
        self.ttl = style_ttl_onnx
        self.dp = style_dp_onnx


class TextToSpeech:
    def __init__(
        self,
        cfgs: dict,
        text_processor: UnicodeProcessor,
        dp_ort: MNNInference,
        text_enc_ort: MNNInference,
        vector_est_ort: MNNInference,
        vocoder_ort: MNNInference,
    ):
        self.cfgs = cfgs
        self.text_processor = text_processor
        self.dp_ort = dp_ort
        self.text_enc_ort = text_enc_ort
        self.vector_est_ort = vector_est_ort
        self.vocoder_ort = vocoder_ort
        self.sample_rate = cfgs["ae"]["sample_rate"]
        self.base_chunk_size = cfgs["ae"]["base_chunk_size"]
        self.chunk_compress_factor = cfgs["ttl"]["chunk_compress_factor"]
        self.ldim = cfgs["ttl"]["latent_dim"]

    def sample_noisy_latent(
        self, duration: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        bsz = len(duration)
        wav_len_max = duration.max() * self.sample_rate
        wav_lengths = (duration * self.sample_rate).astype(np.int64)
        chunk_size = self.base_chunk_size * self.chunk_compress_factor
        latent_len = ((wav_len_max + chunk_size - 1) / chunk_size).astype(np.int32)
        latent_dim = self.ldim * self.chunk_compress_factor
        noisy_latent = np.random.randn(bsz, latent_dim, latent_len).astype(np.float32)
        latent_mask = get_latent_mask(
            wav_lengths, self.base_chunk_size, self.chunk_compress_factor
        )
        noisy_latent = noisy_latent * latent_mask
        return noisy_latent, latent_mask

    def _infer(
        self, text_list: list[str], style: Style, total_step: int, speed: float = 1.05
    ) -> tuple[np.ndarray, np.ndarray, float]:
        assert (
            len(text_list) == style.ttl.shape[0]
        ), "Number of texts must match number of style vectors"
        bsz = len(text_list)
        
        # Start timing for RTF calculation
        start_time = time.time()
        
        text_ids, text_mask = self.text_processor(text_list)
        dur_onnx, *_ = self.dp_ort.run(
            None, {"text_ids": text_ids, "style_dp": style.dp, "text_mask": text_mask}
        )
        dur_onnx = dur_onnx / speed
        text_emb_onnx, *_ = self.text_enc_ort.run(
            None,
            {"text_ids": text_ids, "style_ttl": style.ttl, "text_mask": text_mask},
        )
        xt, latent_mask = self.sample_noisy_latent(dur_onnx)
        total_step_np = np.array([total_step] * bsz, dtype=np.float32)
        for step in range(total_step):
            current_step = np.array([step] * bsz, dtype=np.float32)
            xt, *_ = self.vector_est_ort.run(
                None,
                {
                    "noisy_latent": xt,
                    "text_emb": text_emb_onnx,
                    "style_ttl": style.ttl,
                    "text_mask": text_mask,
                    "latent_mask": latent_mask,
                    "current_step": current_step,
                    "total_step": total_step_np,
                },
            )
        wav, *_ = self.vocoder_ort.run(None, {"latent": xt})
        
        # Calculate elapsed time for RTF
        elapsed_time = time.time() - start_time
        
        return wav, dur_onnx, elapsed_time

    def __call__(
        self,
        text: str,
        style: Style,
        total_step: int,
        speed: float = 1.05,
        silence_duration: float = 0.3,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        assert (
            style.ttl.shape[0] == 1
        ), "Single speaker text to speech only supports single style"
        text_list = chunk_text(text)
        wav_cat = None
        dur_cat = None
        total_elapsed_time = 0.0
        
        for text in text_list:
            wav, dur_onnx, elapsed_time = self._infer([text], style, total_step, speed)
            total_elapsed_time += elapsed_time
            
            if wav_cat is None:
                wav_cat = wav
                dur_cat = dur_onnx
            else:
                silence = np.zeros(
                    (1, int(silence_duration * self.sample_rate)), dtype=np.float32
                )
                wav_cat = np.concatenate([wav_cat, silence, wav], axis=1)
                dur_cat += dur_onnx + silence_duration
                
        # Calculate overall RTF
        total_audio_duration = wav_cat.shape[1] / self.sample_rate
        rtf = total_elapsed_time / total_audio_duration if total_audio_duration > 0 else 0.0
        
        # Print RTF information
        print(f"RTF (Real Time Factor): {rtf:.4f}")
        print(f"Audio Duration: {total_audio_duration:.2f}s")
        print(f"Generation Time: {total_elapsed_time:.2f}s")
        
        return wav_cat, dur_cat, rtf


def get_latent_mask(
    wav_lengths: np.ndarray, base_chunk_size: int, chunk_compress_factor: int
) -> np.ndarray:
    latent_size = base_chunk_size * chunk_compress_factor
    latent_lengths = (wav_lengths + latent_size - 1) // latent_size
    latent_mask = length_to_mask(latent_lengths)
    return latent_mask


def load_mnn(model_path, input_names, output_names):
    return MNNInference(model_path, input_names, output_names)


def load_voice_style(voice_style_paths: list[str], verbose: bool = False) -> Style:
    bsz = len(voice_style_paths)

    # Read first file to get dimensions
    with open(voice_style_paths[0], "r") as f:
        first_style = json.load(f)
    ttl_dims = first_style["style_ttl"]["dims"]
    dp_dims = first_style["style_dp"]["dims"]

    # Pre-allocate arrays with full batch size
    ttl_style = np.zeros([bsz, ttl_dims[1], ttl_dims[2]], dtype=np.float32)
    dp_style = np.zeros([bsz, dp_dims[1], dp_dims[2]], dtype=np.float32)

    # Fill in the data
    for i, voice_style_path in enumerate(voice_style_paths):
        with open(voice_style_path, "r") as f:
            voice_style = json.load(f)

        ttl_data = np.array(
            voice_style["style_ttl"]["data"], dtype=np.float32
        ).flatten()
        ttl_style[i] = ttl_data.reshape(ttl_dims[1], ttl_dims[2])

        dp_data = np.array(voice_style["style_dp"]["data"], dtype=np.float32).flatten()
        dp_style[i] = dp_data.reshape(dp_dims[1], dp_dims[2])

    if verbose:
        print(f"Loaded {bsz} voice styles")
    return Style(ttl_style, dp_style)