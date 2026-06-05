"""Phase D e2e demo — short 2-subject Phantom-Wan generation."""
from phantom_wan_mlx import pipeline_mlx as P

prompt = (
    "A little girl with twin ponytails wearing a light green dress crouches beside blooming "
    "daisies in a sunny meadow; a fluffy brown-and-white dog sits beside her, tongue out, "
    "tail wagging happily."
)
out = P.s2v(
    prompt,
    ["weights/refs/assets/ref1.png", "weights/refs/assets/ref2.png"],
    "outputs/phantom_demo.mp4",
    size=(832, 480), frame_num=17, steps=25, seed=42, verbose=True,
)
print("WROTE", out, flush=True)
