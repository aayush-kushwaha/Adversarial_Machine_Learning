import base64
import io

from torchvision import transforms


to_pil = transforms.ToPILImage()


def tensor_to_pil(tensor):
    if tensor.dim() == 4:
        tensor = tensor[0]
    image = tensor.detach().cpu().clamp(0, 1)
    return to_pil(image)


def pil_to_base64(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")
