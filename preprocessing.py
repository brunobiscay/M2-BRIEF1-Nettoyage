import math
import sys
from datetime import datetime
from os.path import join as join
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import chisquare, norm
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def preprocessing_rappel_rgpd(df):
    print(
        "\n### Penses à vérifier la RGPD (anonymisation des data, consentement, restriction d'accès)\n10 premières lignes des données\n"
    )
    print(df.head(10))

    choix = (
        input(" Es tu conformes RGPD avec ces données ? (o/n) : \n\n").strip().lower()
    )
    if choix == "o":
        pass
    else:
        print("Mauvaise reponse, on stoppe ici")
        sys.exit()


def data_stats(df):
    print("\n# Affichage des statistiques \n\n", df.describe())


def preprocessing_display_missingvalues(df):
    # Affichage des valeurs manquantes
    missing_values = df.isnull().sum()
    print("\n### Affichage des données manquantes par colonne\n")
    print(missing_values, "\n")

    # Demande : suppression des lignes
    choix = (
        input(
            "Souhaitez-vous supprimer les lignes contenant des valeurs manquantes ? (o/n) : "
        )
        .strip()
        .lower()
    )

    if choix == "o":
        df = df.dropna()
        print("Les lignes avec valeurs manquantes ont été supprimées.")
        return df

    print("Aucune suppression effectuée.")

    # Deuxième option : remplacer par la moyenne
    choix2 = (
        input(
            "Souhaitez-vous remplacer les valeurs manquantes par la moyenne des colonnes numériques ? (o/n) : "
        )
        .strip()
        .lower()
    )

    if choix2 == "o":
        colonnes_numeriques = df.select_dtypes(include=["number"]).columns

        for col in colonnes_numeriques:
            if df[col].isna().sum() > 0:
                moyenne = df[col].mean()
                df[col] = df[col].fillna(moyenne)
                print(
                    f"Colonne '{col}' : valeurs manquantes remplacées par la moyenne ({moyenne:.2f})."
                )

        print("Remplacement terminé.")
    else:
        print("Aucun remplacement effectué.")

    return df


def preprocessing_delete_doublons(df):
    # Affichage des valeurs manquantes
    duplicates_keep_true = df.duplicated(keep="first")

    print("\n# Nombre de données dupliquées\n")
    print(duplicates_keep_true.sum(), "\n")

    # Demande : suppression des lignes
    choix = input("Souhaitez-vous supprimer ces doublons ? (o/n) : ").strip().lower()

    if choix == "o":
        df = df.drop_duplicates()
        print("Les lignes avec doublons ont été supprimées.")
        return df

    else:
        print("Aucune suppression effectuée.")

    return df


def preprocessing_suppression_colonnes(df):
    missing_values = df.isnull().sum()
    print(
        "\n\n#Affichage des données manquantes selon colonne d'en-tête \n\n",
        missing_values,
        "\n",
    )
    choix = input("Souhaitez-vous supprimer des colonnes ? (o/n) : ").strip().lower()

    if choix == "o":
        print("\nColonnes disponibles :", ", ".join(df.columns))
        saisie = input(
            "Indiquez les colonnes à supprimer, séparées par une virgule : "
        ).strip()

        # Nettoyage et transformation en liste
        colonnes_a_supprimer = [col.strip() for col in saisie.split(",")]

        # Vérification des colonnes existantes
        colonnes_valides = [col for col in colonnes_a_supprimer if col in df.columns]
        colonnes_invalides = [
            col for col in colonnes_a_supprimer if col not in df.columns
        ]

        if colonnes_invalides:
            print(
                "Attention, ces colonnes n'existent pas :",
                ", ".join(colonnes_invalides),
            )

        if colonnes_valides:
            df = df.drop(columns=colonnes_valides)
            print("Colonnes supprimées :", ", ".join(colonnes_valides))
        else:
            print("Aucune colonne valide à supprimer.")

    else:
        print("Aucune colonne supprimée.")

    return df


def display_distributions(df):
    # Sélection des colonnes numériques
    colonnes = df.select_dtypes(include=["number"]).columns

    if len(colonnes) == 0:
        print("Aucune colonne numérique à afficher.")
        return

    # Définition de la grille
    n_cols = 3  # nombre de graphiques par ligne
    n_rows = math.ceil(len(colonnes) / n_cols)

    plt.figure(figsize=(6 * n_cols, 4 * n_rows))

    for i, col in enumerate(colonnes, 1):
        plt.subplot(n_rows, n_cols, i)
        sns.histplot(df[col])
        plt.title(f"Distribution : {col}")
        plt.xlabel(col)
        plt.ylabel("Fréquence")

    plt.tight_layout()
    plt.show()


def display_outlier_data(df):
    # Sélection des colonnes numériques
    colonnes = df.select_dtypes(include=["number"]).columns

    if len(colonnes) == 0:
        print("Aucune colonne numérique à afficher.")
        return

    # Définition de la grille
    n_cols = 3  # nombre de graphiques par ligne
    n_rows = math.ceil(len(colonnes) / n_cols)

    plt.figure(figsize=(6 * n_cols, 4 * n_rows))

    for i, col in enumerate(colonnes, 1):
        plt.subplot(n_rows, n_cols, i)
        sns.boxplot(df[col])
        plt.title(f"Boxplot (outlier > interquartile (1->3eme quartile)) : {col}")
        plt.xlabel(col)
        plt.ylabel("Fréquence")

    plt.tight_layout()
    plt.show()


def delete_outliers_iqr(df):
    # Détection des colonnes numériques
    colonnes_num = df.select_dtypes(include=["number"]).columns

    colonnes_outliers = []

    # Détection des outliers par colonne
    for col in colonnes_num:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        borne_inf = Q1 - 1.5 * IQR
        borne_sup = Q3 + 1.5 * IQR

        # Vérifier s'il y a des outliers
        if df[(df[col] < borne_inf) | (df[col] > borne_sup)].shape[0] > 0:
            colonnes_outliers.append(col)

    # Affichage des colonnes concernées
    print("\n### Colonnes contenant des outliers (méthode IQR) :")
    if len(colonnes_outliers) == 0:
        print("Aucune colonne ne contient d'outliers détectés.")
        return df
    else:
        print(", ".join(colonnes_outliers))

    # Demande à l'utilisateur
    choix = (
        input(
            "\nSouhaitez-vous supprimer les lignes contenant des outliers pour ces colonnes ? (o/n) : "
        )
        .strip()
        .lower()
    )

    if choix == "o":
        # Application du filtre IQR sur toutes les colonnes concernées
        df_clean = df.copy()
        for col in colonnes_outliers:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            borne_inf = Q1 - 1.5 * IQR
            borne_sup = Q3 + 1.5 * IQR

            df_clean = df_clean[
                (df_clean[col] >= borne_inf) & (df_clean[col] <= borne_sup)
            ]

        print("\nLes lignes contenant des outliers ont été supprimées.")
        print(
            f"Nouvelle taille du DataFrame : {df_clean.shape} vs ancienne taille {df.shape}"
        )

        return df_clean

    else:
        print("Aucune suppression effectuée.")
        return df


def displayanddrop_correlation_data(df):
    numerical_df = df.select_dtypes(include=np.number)

    # Calculate the correlation matrix
    correlation_matrix = numerical_df.corr()

    print("\n\nBut : supprimer les colonnes de données selon seuil de correlation\n")
    print("Matrices de correlation et heatmap associée\n")
    print(correlation_matrix, "\n")

    # Plot the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        correlation_matrix,
        annot=True,  # Affiche les valeurs
        fmt=".2f",  # Format à 2 décimales
        cmap="coolwarm",  # Palette de couleurs
        linewidths=0.5,  # Lignes entre les cases
        square=True,  # Cases carrées
    )

    plt.title("Heatmap des corrélations")
    plt.tight_layout()
    plt.show()

    # Use absolute correlation
    abs_corr_matrix = correlation_matrix.abs()
    threshold = 0.9
    # Get the upper triangle of the correlation matrix (excluding the diagonal)

    upper_triangle = abs_corr_matrix.where(
        np.triu(np.ones(abs_corr_matrix.shape), k=1).astype(bool)
    )

    # Find features with correlation above the threshold
    to_drop = set()  # Use a set to avoid duplicates
    for column in upper_triangle.columns:
        highly_correlated_with = upper_triangle.index[
            upper_triangle[column] > threshold
        ].tolist()

    if highly_correlated_with:
        # Basic strategy: Drop the current column if it's highly correlated with any previous one
        # More sophisticated strategies could be implemented here
        # For simplicity, let's collect the column name itself if it correlates highly
        # with any feature already checked (index < column)
        if any(upper_triangle[column] > threshold):
            to_drop.add(column)  # Example: Add the second feature in the pair

    print(f"Features to drop (based on default threshold {threshold}): {list(to_drop)}")

    # Drop the selected features from the original DataFrame
    df_reduced = df.drop(columns=list(to_drop))
    print(f"Nombre original de caracteristiques: {df.shape[1]}")
    print(
        f"Nombre de  caracteristiques après le filtrage de correlation à {threshold}: {df_reduced.shape[1]}"
    )

    return df_reduced


def eval_loi_normale(df, alpha=0.05, bins=10):
    """
    Teste la normalité de chaque colonne numérique d'un DataFrame avec un test Chi².
    Applique StandardScaler si normal, sinon MinMaxScaler.

    Parameters
    ----------
    df : pandas.DataFrame
        Le DataFrame contenant les colonnes à scaler.
    alpha : float
        Seuil de significativité pour le test Chi².
    bins : int
        Nombre de classes pour l'histogramme utilisé dans le test Chi².

    Returns
    -------
    scaled_df : pandas.DataFrame
        Le DataFrame transformé.
    scalers : dict
        Dictionnaire {colonne: scaler utilisé}.
    normality_results : dict
        Dictionnaire {colonne: p-value du test Chi²}.
    """

    numeric_cols = df.select_dtypes(include=np.number).columns
    scaled_df = df.copy()
    scalers = {}
    p_values = {}

    for col in numeric_cols:
        data = df[col].dropna()

        # Histogramme observé
        observed, bin_edges = np.histogram(data, bins=bins)

        # Distribution normale théorique
        mu, sigma = data.mean(), data.std()
        cdf = norm.cdf(bin_edges, loc=mu, scale=sigma)
        expected = len(data) * np.diff(cdf)

        # Évite les zéros
        expected[expected == 0] = 1e-6

        # 🔧 Correction essentielle : normaliser expected
        expected = expected * (observed.sum() / expected.sum())

        # Test Chi²
        chi2_stat, p_value = chisquare(f_obs=observed, f_exp=expected)
        p_values[col] = p_value

        # Choix du scaler
        scaler = StandardScaler() if p_value > alpha else MinMaxScaler()
        scalers[col] = scaler
        scaled_df[col] = scaler.fit_transform(df[[col]])

    return scaled_df, scalers, p_values


def sauvegarde_df_nen_csv(df):
    """
    Sauvegarde un DataFrame dans un dossier 'data' situé
    au même endroit que le script qui appelle cette fonction.
    Le fichier est nommé avec la date (YYYY-MM-DD.csv).
    """
    # Répertoire du script
    script_dir = Path(__file__).resolve().parent

    # Dossier data relatif au script
    data_dir = script_dir / "dataclean"
    data_dir.mkdir(exist_ok=True)

    # Nom de fichier basé sur la date
    filename = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".csv"
    filepath = data_dir / filename

    # Sauvegarde du DataFrame
    df.to_csv(filepath, index=False)

    return filepath
