from utils.run_utils import get_model


if __name__ == "__main__":
    model = get_model("med_3d_llm", "med_3d_llm_params_brats_met_eval.yml")
    model.setup()
    model.evaluate()
