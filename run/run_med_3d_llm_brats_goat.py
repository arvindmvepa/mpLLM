from utils.run_utils import get_model


if __name__ == "__main__":
    model = get_model("med_3d_llm", "med_3d_llm_params_brats_goat.yml")
    model.setup()
    model.run()
    model.evaluate()
