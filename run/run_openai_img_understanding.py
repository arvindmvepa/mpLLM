from utils.run_utils import get_model


if __name__ == "__main__":
    model = get_model("openai_llm", "openai_image_understanding.yml")
    model.setup()
    model.run()