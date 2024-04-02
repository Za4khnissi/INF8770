import argparse
import numpy as np
import pandas as pd


def get_parser():
    parser = argparse.ArgumentParser(description="Evaluation de INF8770 - TP3")

    parser.add_argument('--file', type=str, help="Nom du fichier évalué")
    parser.add_argument('--file_gt', type=str, help="Nom du fichier contenant les solutions")

    return parser


def evaluate(file, file_gt):
    """
    Evalue les performances du système de reconnaissance de vidéos
    :param file: chemin d'accès au fichier des résultats de l'algorithme
    :param file_gt: chemin d'accès au fichier contenant les solutions
    """
    
    prediction = pd.read_csv(file)
    solution = pd.read_csv(file_gt)

    assert solution.shape == (1000, 3), f"Erreur: le fichier gt.csv est corrompu !"
    assert prediction.shape == solution.shape, f"Erreur: votre fichier n'a pas les bonnes dimensions : {prediction.shape} != {solution.shape}"
    assert all(prediction.columns == ['image', 'video_pred', 'minutage_pred']), f"Renommez les colonnes de votre fichier en ['image', 'video_pred', 'minutage_pred']"

    df_merge = pd.merge(solution, prediction, on='image')

    # Nombre de prédictions correctes
    n_correct = np.sum(df_merge['video'] == df_merge['video_pred'])
    pct = 100 * n_correct / df_merge.shape[0]

    # Ecart temporel
    all_gaps = [abs(mn - pred_mn) for (mn, pred_mn, video, video_pred) in zip(df_merge.minutage, df_merge.minutage_pred, df_merge.video, df_merge.video_pred) if (video == video_pred) & (video != 'out')]
    gap = np.mean(all_gaps) if all_gaps else np.nan
        
    return pct, gap, n_correct, df_merge.shape[0]


if __name__ == '__main__':

    args = get_parser().parse_args()
    pct, gap, n_correct, n_images = evaluate(file=args.file, file_gt=args.file_gt)

    print(f"Taux de bonnes réponses : {pct:0.1f}% ({n_correct}/{n_images})")
    print(f"Ecart temporel moyen : {gap:0.2f} sec")