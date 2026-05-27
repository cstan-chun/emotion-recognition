import argparse
import json


from config import load_config
from pipeline import EmotionRecognitionPipeline
from output import OutputGenerator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--video", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    cfg = load_config(args.config)
    pipeline = EmotionRecognitionPipeline(cfg)

    print(f"Processing: {args.video}")
    result = pipeline.process(args.video, checkpoint_path=args.checkpoint)

    gen = OutputGenerator(cfg)
    video_out, json_out = gen.generate_all(args.video, result, args.output_dir)

    print(f"Annotated video: {video_out}")
    print(f"Result JSON: {json_out}")
    print(f"Summary: {json.dumps(result['summary'], indent=2)}")


if __name__ == "__main__":
    main()
