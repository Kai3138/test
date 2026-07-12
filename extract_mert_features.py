import os
import json
import argparse
import numpy as np
import librosa
import torch
from tqdm import tqdm
from transformers import Wav2Vec2FeatureExtractor, AutoModel


def align_length(feat, target_len):
    if len(feat) == target_len:
        return feat.astype(np.float32)

    src_x = np.linspace(0, 1, len(feat))
    tgt_x = np.linspace(0, 1, target_len)

    aligned = np.stack([
        np.interp(tgt_x, src_x, feat[:, i])
        for i in range(feat.shape[1])
    ], axis=1)

    return aligned.astype(np.float32)


def load_base_length(base_feat_dir, music_id):
    path = os.path.join(base_feat_dir, music_id + ".json")
    with open(path, "r") as f:
        data = json.load(f)
    return len(data["music_array"])


def extract_mert(audio_path, processor, model, device, layer_mode):
    audio, sr = librosa.load(audio_path, sr=24000, mono=True)

    inputs = processor(
        audio,
        sampling_rate=24000,
        return_tensors="pt",
        padding=True
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    hidden_states = outputs.hidden_states

    if layer_mode == "last4_mean":
        feat = torch.stack(hidden_states[-4:], dim=0).mean(0)
    elif layer_mode == "last4_cat":
        feat = torch.cat(hidden_states[-4:], dim=-1)
    else:
        feat = hidden_states[-1]

    return feat.squeeze(0).cpu().numpy().astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_video_dir", type=str, default="aist_plusplus_final/all_musics")
    parser.add_argument("--store_dir", type=str, default="data/aistpp_music_feat_7.5fps_mert")
    parser.add_argument("--base_feat_dir", type=str, default="data/aistpp_music_feat_7.5fps")
    parser.add_argument("--mert_model", type=str, default="m-a-p/MERT-v1-95M")
    parser.add_argument("--layer_mode", type=str, default="last", choices=["last", "last4_mean", "last4_cat"])
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    os.makedirs(args.store_dir, exist_ok=True)

    processor = Wav2Vec2FeatureExtractor.from_pretrained(args.mert_model,trust_remote_code=True)
    model = AutoModel.from_pretrained(args.mert_model, output_hidden_states=True,trust_remote_code=True)
    model = model.to(args.device).eval()

    audio_fnames = sorted(os.listdir(args.input_video_dir))

    for audio_fname in tqdm(audio_fnames):
        if not audio_fname.endswith((".wav", ".mp3")):
            continue

        music_id = os.path.splitext(audio_fname)[0]
        audio_path = os.path.join(args.input_video_dir, audio_fname)

        mert_feat = extract_mert(
            audio_path,
            processor,
            model,
            args.device,
            args.layer_mode
        )

        target_len = load_base_length(args.base_feat_dir, music_id)
        mert_feat = align_length(mert_feat, target_len)

        save_path = os.path.join(args.store_dir, music_id + ".json")
        with open(save_path, "w") as f:
            json.dump({
                "id": music_id,
                "music_array": mert_feat.tolist()
            }, f)

        print(music_id, mert_feat.shape)


if __name__ == "__main__":
    main()