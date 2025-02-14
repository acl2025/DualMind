# Load model directly
from transformers import AutoModel
model = AutoModel.from_pretrained("THUDM/chatglm3-6b", trust_remote_code=True)