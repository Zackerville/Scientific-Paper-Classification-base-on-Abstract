from typing import List, Literal, Optional
import numpy as np

Mode = Literal['query', 'passage', 'raw']

class EmbeddingVectorizer:
    def __init__(
        self,
        model_name: str = 'intfloat/multilingual-e5-base',
        normalize: bool = True,
        device: Optional[str] = None,
        batch_size: int = 32,
        show_progress: bool = False,
    ):
        self.model_name = model_name
        self.normalize = normalize
        self.device = device
        self.batch_size = batch_size
        self.show_progress = show_progress
        self.model = None

    def _ensure_model(self):
        if self.model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except Exception as e:
                raise ImportError(
                    "Cần cài sentence-transformers và torch trong môi trường đang chạy. "
                    "pip install --index-url https://download.pytorch.org/whl/cpu torch "
                    "&& pip install sentence-transformers"
                ) from e
            self.model = SentenceTransformer(self.model_name, device=self.device)

    def _format_inputs(self, texts: List[str], mode: Mode) -> List[str]:
        if mode not in {'query', 'passage', 'raw'}:
            raise ValueError("mode phải là 'query', 'passage' hoặc 'raw'")
        if mode == 'raw':
            return [t.strip() for t in texts]
        return [f'{mode}: {t.strip()}' for t in texts]

    def transform_numpy(self, texts: List[str], mode: Mode = 'query') -> np.ndarray:
        self._ensure_model()
        inputs = self._format_inputs(texts, mode)
        emb = self.model.encode(
            inputs,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
            batch_size=self.batch_size,
            show_progress_bar=self.show_progress,
        )
        return emb  # np.ndarray (n_samples, dim)

    # nếu vẫn muốn API trả list[list[float]]:
    def transform(self, texts: List[str], mode: Mode = 'query'):
        return self.transform_numpy(texts, mode).tolist()
