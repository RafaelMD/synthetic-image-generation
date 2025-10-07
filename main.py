import argparse
import os

from dotenv import load_dotenv

from synthetic_crew.pipeline import SyntheticImageGenerationCrew


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Synthetic Image Generation orchestrated with CrewAI agents",
    )
    parser.add_argument("-e", "--entity", type=str, required=True, help="Entity to add in the images")
    parser.add_argument(
        "-c",
        "--context_limit",
        type=int,
        default=3,
        help="The limit for generate contexts",
    )
    parser.add_argument(
        "-i",
        "--input_folder",
        type=str,
        default="input_images",
        help="Input folder with images",
    )
    parser.add_argument(
        "-o",
        "--output_folder",
        type=str,
        default="output_images",
        help="Output folder for generated images",
    )
    parser.add_argument(
        "-d",
        "--discard_folder",
        type=str,
        default="discarded_images",
        help="Folder for discarded images",
    )
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()

    api_key = os.getenv("API_KEY")
    if not api_key:
        raise RuntimeError("Missing API_KEY environment variable.")

    discard_folder = os.path.join(args.discard_folder, args.entity)

    pipeline = SyntheticImageGenerationCrew(
        api_key=api_key,
        entity=args.entity,
        context_limit=args.context_limit,
        input_folder=args.input_folder,
        output_folder=args.output_folder,
        discard_folder=discard_folder,
    )

    pipeline.clean_discard_folder()
    report = pipeline.run()

    report_path = os.path.join(pipeline.entity_output_folder, "report.json")
    print(f"Report saved in {report_path}")
    print(f"Total processing time: {report['processing_time']}")


if __name__ == "__main__":
    main()
