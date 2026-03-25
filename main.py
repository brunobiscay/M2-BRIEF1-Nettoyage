from os.path import join as join

import pandas as pd

from preprocessing import (
    data_stats,
    delete_outliers_iqr,
    display_distributions,
    display_outlier_data,
    displayanddrop_correlation_data,
    eval_loi_normale,
    preprocessing_delete_doublons,
    preprocessing_display_missingvalues,
    preprocessing_rappel_rgpd,
    preprocessing_suppression_colonnes,
    sauvegarde_df_nen_csv,
)


def main():
    print("Etape 1.Exploration des données, et profilage des données\n\n")

    data = pd.read_csv(
        join("data", "fichier-de-donnees-numeriques-69202f25dea8b267811864.csv")
    )

    preprocessing_rappel_rgpd(data)

    data_stats(data)

    display_distributions(data)
    data_step1 = displayanddrop_correlation_data(data)

    print(
        "Etape 2.Detection des pbms,\n2.Nettoyage/correction/suppr des valeurs erronées\n\n"
    )

    data_step2 = preprocessing_display_missingvalues(data_step1)

    data_step3 = preprocessing_delete_doublons(data_step2)

    data_step4 = preprocessing_suppression_colonnes(data_step3)
    print("Colonnes restantes :", ", ".join(data_step3.columns))

    display_outlier_data(data_step4)

    data_step5 = delete_outliers_iqr(data_step4)

    print("\n\nEtape 3.  La standardisation et la normalisation (Min-Max Scaling\n")
    print(
        "Si notre caractéristique numérique suit une loi approximativement normale, on standardise. Dans les autres cas, on normalise"
    )

    data_step6, scalers, pvals = eval_loi_normale(data_step5)

    print("\nP-values du test Chi² :", pvals)
    print("Scalers utilisés :", {k: type(v).__name__ for k, v in scalers.items()})

    # data_step6 = normalisation_data(data_step5)

    data_stats(data_step6)

    print("Etape 4 : La validation\n\n")

    filepath = sauvegarde_df_nen_csv(data_step6)
    print("Fichier créé avec données nettoyées :", filepath)


if __name__ == "__main__":
    main()
