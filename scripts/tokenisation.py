import spacy
import json
import re
import argparse
import time
from pathlib import Path
from collections import Counter
import sys
import os

# Ajouter le dossier parent au path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config_loader import ConfigLoader
from utils.logger import setup_logging, get_logger

# Initialiser le logger
logger = get_logger(__name__)


class BatchTokenizer:
    """
    Classe pour tokeniser en batch tous les fichiers BSV nettoyés
    """
    
    def __init__(self, force_reprocess=False, target_year=None):
        """
        Initialise le tokenizer batch
        
        Args:
            force_reprocess: Si True, re-traite les fichiers déjà traités
            target_year: Si spécifié, ne traite que cette année
        """
        self.force_reprocess = force_reprocess
        self.target_year = target_year
        
        # Charger la config
        self.config_loader = ConfigLoader("config.yaml")
        self.base_dir = Path(self.config_loader.base_dir)
        
        # Chemins
        self.processed_dir = self.base_dir / self.config_loader.config["data"]["processed_dir"]
        self.results_dir = self.base_dir / self.config_loader.config["data"]["results_dir"]
        self.source_dir = self.processed_dir / "clean_txt" / "bourgogne_franche_comte"
        self.output_dir = self.results_dir / "tokens"
        
        # Créer le dossier de sortie
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Charger SpaCy
        logger.info("Chargement du modele SpaCy...")
        try:
            self.nlp = spacy.load("fr_core_news_lg")
        except:
            logger.error("Modele fr_core_news_lg non trouve. Installation necessaire:")
            logger.error("python -m spacy download fr_core_news_lg")
            sys.exit(1)
        
        # Statistiques globales
        self.global_stats = {
            'total_files': 0,
            'processed_files': 0,
            'skipped_files': 0,
            'failed_files': 0,
            'total_tokens': 0,
            'unique_tokens': set(),
            'processing_time': 0,
            'errors': []
        }
        
        # Compteur global de tokens
        self.global_token_counter = Counter()
    
    def _preprocess_text(self, text):
        """
        Pré-traite le texte avant tokenisation
        
        Args:
            text: Texte brut
            
        Returns:
            str: Texte pré-traité
        """
        # Garder "25%" comme un seul token
        text = re.sub(r'(\d+)\s*%', r'\1%', text)
        # Fusionner les chiffres séparés
        text = re.sub(r'(\d+)\s+(\d+)', r'\1\2', text)
        return text
    
    def _extract_tokens(self, doc):
        """
        Extrait les tokens lemmatisés d'un document SpaCy
        
        Args:
            doc: Document SpaCy
            
        Returns:
            tuple: (liste de tokens, compteur de fréquences)
        """
        tokens_lemmatises = []
        token_counter = Counter()
        
        for token in doc:
            # Sauter les éléments non désirés
            if token.is_space or token.is_punct or token.is_stop:
                continue
            
            # Cas 1: Nombres et pourcentages (garder tels quels)
            if token.like_num or '%' in token.text:
                token_text = token.text
                tokens_lemmatises.append(token_text)
                token_counter[token_text] += 1
                continue
            
            # Cas 2: Mots alphabétiques (lemmatiser)
            if token.is_alpha:
                lemme = token.lemma_.lower().strip()
                if lemme and len(lemme) > 1:  # Éviter lettres seules
                    tokens_lemmatises.append(lemme)
                    token_counter[lemme] += 1
        
        return tokens_lemmatises, token_counter
    
    def _extract_bigrams(self, tokens_bruts):
        """
        Extrait tous les bigrams pertinents avec leurs fréquences
        
        Args:
            tokens_bruts: Liste des tokens non lemmatisés
            
        Returns:
            list: Liste de dictionnaires {bigram, frequency}
        """
        bigrams_counter = Counter()
        
        # Mots à éviter dans les bigrams
        stop_words = {'de', 'du', 'la', 'le', 'et', 'à', 'des', 'les', 'un', 'une'}
        
        for i in range(len(tokens_bruts) - 1):
            mot1 = tokens_bruts[i]
            mot2 = tokens_bruts[i + 1]
            
            # Filtrer les bigrams non pertinents
            if (len(mot1) > 2 and len(mot2) > 2 and
                mot1.lower() not in stop_words and
                mot2.lower() not in stop_words):
                
                bigram = f"{mot1} {mot2}"
                
                # Vérifier si c'est un terme technique
                doc_bigram = self.nlp(bigram)
                if len(doc_bigram) == 2:
                    # Garder si les deux tokens sont des noms/adj/verbes
                    if (doc_bigram[0].pos_ in ['NOUN', 'PROPN', 'VERB', 'ADJ'] and
                        doc_bigram[1].pos_ in ['NOUN', 'PROPN', 'ADJ']):
                        bigrams_counter[bigram] += 1
        
        # Convertir en liste de dictionnaires triée par fréquence
        bigrams_list = [
            {"bigram": bigram, "frequency": freq}
            for bigram, freq in bigrams_counter.most_common()
        ]
        
        return bigrams_list
    
    def _extract_date_from_filename(self, filename):
        """
        Extrait la date du nom de fichier BSV
        
        Args:
            filename: Nom du fichier (ex: bsv_gc_n_1_du_23_08_22.txt)
            
        Returns:
            str: Date au format YYYY-MM ou "inconnue"
        """
        date_match = re.search(r'(\d{2})_(\d{2})_(\d{2,4})', filename)
        if date_match:
            jour, mois, annee = date_match.groups()
            # Convertir année courte en longue
            if len(annee) == 2:
                annee = f"20{annee}"
            return f"{annee}-{mois}"
        return "inconnue"
    
    def process_file(self, file_path, year):
        """
        Traite un seul fichier BSV
        
        Args:
            file_path: Chemin vers le fichier texte
            year: Année du fichier
            
        Returns:
            dict: Résultat de la tokenisation ou None si erreur
        """
        try:
            # Lire le fichier
            with open(file_path, 'r', encoding='utf-8') as f:
                texte = f.read()
            
            if not texte.strip():
                logger.warning(f"Fichier vide: {file_path.name}")
                return None
            
            # Pré-traitement
            texte = self._preprocess_text(texte)
            
            # Analyse SpaCy
            doc = self.nlp(texte)
            
            # Extraction des tokens (lemmatisés et bruts)
            tokens_lemmatises, token_counter = self._extract_tokens(doc)
            
            # Extraction des tokens bruts pour les bigrams
            tokens_bruts = [token.text for token in doc if not token.is_space]
            
            # Suppression des doublons consécutifs dans les tokens lemmatisés
            tokens_filtres = []
            for i in range(len(tokens_lemmatises)):
                if i == 0 or tokens_lemmatises[i] != tokens_lemmatises[i-1]:
                    tokens_filtres.append(tokens_lemmatises[i])
            
            # Tokens uniques ordonnés
            tokens_uniques = []
            vus = set()
            for token in tokens_filtres:
                if token not in vus:
                    tokens_uniques.append(token)
                    vus.add(token)
            
            # Extraction des bigrams
            bigrams = self._extract_bigrams(tokens_bruts)
            
            # Date du fichier
            date_bsv = self._extract_date_from_filename(file_path.stem)
            
            # Métadonnées
            file_size = file_path.stat().st_size
            
            # Construction du résultat
            result = {
                "filename": file_path.name,
                "year": year,
                "tokens_lemmatises": tokens_uniques,
                "token_frequencies": dict(token_counter.most_common(100)),  # Top 100 pour alléger
                "bigrams": bigrams,
                "metadata": {
                    "nb_mots_originaux": len(tokens_lemmatises),
                    "nb_tokens_uniques": len(tokens_uniques),
                    "nb_bigrams": len(bigrams),
                    "date_bsv": date_bsv,
                    "date_extraction": time.strftime("%Y-%m-%d"),
                    "file_size_bytes": file_size,
                    "chemin_original": str(file_path)
                }
            }
            
            # Mise à jour des stats globales
            self.global_stats['total_tokens'] += len(tokens_lemmatises)
            self.global_stats['unique_tokens'].update(tokens_uniques)
            self.global_token_counter.update(token_counter)
            
            return result
            
        except Exception as e:
            logger.error(f"Erreur lors du traitement de {file_path.name}: {e}")
            self.global_stats['errors'].append({
                'file': str(file_path),
                'error': str(e)
            })
            return None
    
    def process_all_files(self):
        """
        Traite tous les fichiers BSV du dossier source
        """
        logger.info("Debut du traitement batch des fichiers BSV")
        logger.info(f"Dossier source: {self.source_dir}")
        logger.info(f"Dossier sortie: {self.output_dir}")
        
        if not self.source_dir.exists():
            logger.error(f"Dossier source inexistant: {self.source_dir}")
            return
        
        start_time = time.time()
        
        # Parcourir les années
        for year_dir in sorted(self.source_dir.iterdir()):
            if not year_dir.is_dir():
                continue
            
            year = year_dir.name
            
            # Filtrer par année si spécifié
            if self.target_year and year != str(self.target_year):
                continue
            
            logger.info(f"Traitement de l'annee {year}")
            
            # Créer le dossier de sortie pour cette année
            output_year_dir = self.output_dir / year
            output_year_dir.mkdir(parents=True, exist_ok=True)
            
            # Traiter chaque fichier .txt
            txt_files = list(year_dir.glob("*.txt"))
            self.global_stats['total_files'] += len(txt_files)
            
            for file_path in txt_files:
                output_path = output_year_dir / f"{file_path.stem}.json"
                
                # Vérifier si déjà traité
                if output_path.exists() and not self.force_reprocess:
                    logger.debug(f"Fichier deja traite, ignore: {file_path.name}")
                    self.global_stats['skipped_files'] += 1
                    continue
                
                # Traiter le fichier
                logger.info(f"Traitement: {file_path.name}")
                result = self.process_file(file_path, year)
                
                if result:
                    # Sauvegarder le JSON
                    with open(output_path, 'w', encoding='utf-8') as f:
                        json.dump(result, f, ensure_ascii=False, indent=2)
                    
                    self.global_stats['processed_files'] += 1
                    logger.debug(f"JSON sauvegarde: {output_path.name}")
                else:
                    self.global_stats['failed_files'] += 1
        
        # Temps total
        self.global_stats['processing_time'] = time.time() - start_time
        
        # Générer le rapport
        self._generate_report()
    
    def _generate_report(self):
        """
        Génère un rapport de synthèse de la tokenisation
        """
        logger.info("Generation du rapport de synthese")
        
        # Préparer le rapport
        report = {
            "date_generation": time.strftime("%Y-%m-%d %H:%M:%S"),
            "statistiques": {
                "total_fichiers": self.global_stats['total_files'],
                "fichiers_traites": self.global_stats['processed_files'],
                "fichiers_ignores": self.global_stats['skipped_files'],
                "fichiers_en_erreur": self.global_stats['failed_files'],
                "tokens_totaux": self.global_stats['total_tokens'],
                "tokens_uniques": len(self.global_stats['unique_tokens']),
                "temps_traitement_secondes": round(self.global_stats['processing_time'], 2)
            },
            "top_50_tokens_globaux": [
                {"token": token, "frequency": freq}
                for token, freq in self.global_token_counter.most_common(50)
            ],
            "erreurs": self.global_stats['errors']
        }
        
        # Sauvegarder le rapport
        report_path = self.output_dir / "tokenization_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Rapport sauvegarde: {report_path}")
        
        # Afficher le résumé
        logger.info("=" * 80)
        logger.info("TRAITEMENT BATCH TERMINE")
        logger.info("=" * 80)
        logger.info(f"Fichiers traites: {report['statistiques']['fichiers_traites']}/{report['statistiques']['total_fichiers']}")
        logger.info(f"Fichiers ignores: {report['statistiques']['fichiers_ignores']}")
        logger.info(f"Fichiers en erreur: {report['statistiques']['fichiers_en_erreur']}")
        logger.info(f"Tokens totaux: {report['statistiques']['tokens_totaux']}")
        logger.info(f"Tokens uniques: {report['statistiques']['tokens_uniques']}")
        logger.info(f"Temps de traitement: {report['statistiques']['temps_traitement_secondes']}s")
        logger.info("=" * 80)
        
        # Afficher les top tokens
        logger.info("Top 10 tokens les plus frequents:")
        for i, item in enumerate(report['top_50_tokens_globaux'][:10], 1):
            logger.info(f"  {i}. {item['token']}: {item['frequency']} occurrences")


def main():
    """
    Fonction principale avec arguments en ligne de commande
    """
    parser = argparse.ArgumentParser(
        description="Tokenisation batch des fichiers BSV"
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help="Force le re-traitement des fichiers deja traites"
    )
    parser.add_argument(
        '--year',
        type=int,
        help="Traiter uniquement une annee specifique (ex: 2022)"
    )
    
    args = parser.parse_args()
    
    # Initialiser le logging
    setup_logging()
    logger.info("Demarrage du traitement batch de tokenisation")
    logger.info("Plant Health NLP Analysis - Polytech Dijon")
    
    try:
        tokenizer = BatchTokenizer(
            force_reprocess=args.force,
            target_year=args.year
        )
        tokenizer.process_all_files()
        
        logger.info("Script termine avec succes")
        return 0
        
    except KeyboardInterrupt:
        logger.warning("Interruption par l'utilisateur (Ctrl+C)")
        return 1
        
    except Exception as e:
        logger.error("Erreur fatale lors de l'execution")
        logger.exception(e)
        return 1


if __name__ == "__main__":
    sys.exit(main())