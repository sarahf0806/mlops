import argparse

from model_pipeline import (
    prepare_data,
    train_model,
    evaluate_model,
    save_model,
    load_model,
)


def main():

    parser = argparse.ArgumentParser(
        description="Pipeline de classification pour la potabilité de l'eau."
    )

    parser.add_argument(
        "step",
        type=str,
        help="Étape : prepare | train | train_and_save | evaluate",
    )

    parser.add_argument(
        "--data",
        type=str,
        default="water_potability.csv",
        help="Chemin vers dataset",
    )

    parser.add_argument(
        "--model_path",
        type=str,
        default="rf_model.joblib",
        help="Chemin modèle",
    )

    args = parser.parse_args()

    # -----------------------
    # STEP : PREPARE
    # -----------------------
    if args.step == "prepare":
        print(">> Préparation des données...")
        X_train, X_test, y_train, y_test = prepare_data(args.data)
        print("Shapes :")
        print("X_train :", X_train.shape)
        print("X_test  :", X_test.shape)
        print("y_train :", y_train.shape)
        print("y_test  :", y_test.shape)

    # -----------------------
    # STEP : TRAIN
    # -----------------------
    elif args.step == "train":
        print(">> Préparation des données...")
        X_train, X_test, y_train, y_test = prepare_data(args.data)

        print(">> Entraînement du modèle...")
        model = train_model(X_train, y_train)

        print(">> Évaluation...")
        evaluate_model(model, X_test, y_test)

    # -----------------------
    # STEP : TRAIN + SAVE
    # -----------------------
    elif args.step == "train_and_save":
        print(">> Préparation des données...")
        X_train, X_test, y_train, y_test = prepare_data(args.data)

        print(">> Entraînement du modèle...")
        model = train_model(X_train, y_train)

        print(">> Sauvegarde du modèle...")
        save_model(model, args.model_path)

        print(">> Évaluation...")
        evaluate_model(model, X_test, y_test)

    # -----------------------
    # STEP : EVALUATE EXISTING MODEL
    # -----------------------
    elif args.step == "evaluate":
        print(">> Préparation des données...")
        _, X_test, _, y_test = prepare_data(args.data)

        print(">> Chargement du modèle sauvegardé...")
        model = load_model(args.model_path)

        print(">> Évaluation...")
        evaluate_model(model, X_test, y_test)

    else:
        print("Étape inconnue.")


if __name__ == "__main__":
    main()
