import spacy
from spacy.matcher import PhraseMatcher, Matcher
import json
import argparse
import time
from pathlib import Path
from collections import Counter, defaultdict
import sys
import os
import re

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config_loader import ConfigLoader
from utils.logger import setup_logging, get_logger

logger = get_logger(__name__)


class EntityExtractor:
    """
    Classe pour extraire les entités nommées (cultures, pathogènes, ravageurs, symptômes)
    avec matching intelligent et gestion des variations
    """
    
    def __init__(self, force_reprocess=False, target_year=None):
        """
        Initialise l'extracteur d'entités
        
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
        self.output_dir = self.results_dir / "entities"
        self.list_dir = self.base_dir / "list"
        
        # Créer le dossier de sortie
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Charger SpaCy
        logger.info("Chargement du modele SpaCy...")
        try:
            self.nlp = spacy.load("fr_core_news_lg")
        except:
            logger.error("Modele fr_core_news_lg non trouve")
            sys.exit(1)
        
        # Charger les dictionnaires
        self.dictionaries = self._load_dictionaries()
        
        # Créer les structures de matching
        self.matchers = self._create_matchers()
        self.keywords = self._extract_keywords()
        
        # Statistiques globales
        self.global_stats = {
            'total_files': 0,
            'processed_files': 0,
            'skipped_files': 0,
            'failed_files': 0,
            'total_entities': defaultdict(int),
            'unique_entities': defaultdict(set),
            'processing_time': 0,
            'errors': []
        }
        
        # Compteurs globaux
        self.global_entity_counter = defaultdict(Counter)
        self.global_cooccurrences = defaultdict(Counter)
    
    def _load_dictionaries(self):
        """
        Charge tous les dictionnaires de termes
        
        Returns:
            dict: Dictionnaires chargés par type
        """
        logger.info("Chargement des dictionnaires...")
        
        dictionaries = {
            'cultures': [],
            'pathogenes': [],
            'ravageurs': [],
            'symptomes': []
        }
        
        # Charger depuis les fichiers .txt
        for dict_type in ['cultures', 'pathogenes', 'ravageurs', 'symptomes']:
            file_path = self.list_dir / f"{dict_type}.txt"
            if file_path.exists():
                with open(file_path, 'r', encoding='utf-8') as f:
                    terms = [line.strip().lower() for line in f if line.strip()]
                    dictionaries[dict_type] = terms
                    logger.info(f"  {dict_type}: {len(terms)} termes charges")
            else:
                logger.warning(f"Fichier {dict_type}.txt non trouve")
        
        # Charger aussi depuis le JSON si disponible
        json_path = self.list_dir / "heathPlantDictionnary.json"
        if json_path.exists():
            with open(json_path, 'r', encoding='utf-8') as f:
                json_dict = json.load(f)
                
                # Fusionner avec les listes existantes
                for key in ['cultures', 'pathogenes', 'ravageurs', 'symptomes']:
                    if key in json_dict:
                        json_terms = [term.lower() for term in json_dict[key]]
                        # Éviter les doublons
                        existing_set = set(dictionaries[key])
                        new_terms = [t for t in json_terms if t not in existing_set]
                        dictionaries[key].extend(new_terms)
                        logger.info(f"  {key}: +{len(new_terms)} termes du JSON")
        
        return dictionaries
    
    def _extract_keywords(self):
        """
        Extrait les mots-clés principaux de chaque terme pour matching partiel
        
        Returns:
            dict: Mots-clés par type d'entité
        """
        logger.info("Extraction des mots-cles pour matching intelligent...")
        
        keywords = {
            'cultures': {},
            'pathogenes': {},
            'ravageurs': {},
            'symptomes': {}
        }
        
        # Mots à ignorer dans l'extraction de mots-clés
        stop_words = {'de', 'du', 'des', 'la', 'le', 'les', 'un', 'une', 
                      'd', 'l', 'à', 'au', 'aux', 'et', 'ou'}
        
        for entity_type, terms in self.dictionaries.items():
            for term in terms:
                # Analyser le terme avec SpaCy
                doc = self.nlp(term)
                
                # Extraire les mots significatifs
                significant_words = []
                for token in doc:
                    if (token.is_alpha and 
                        token.text.lower() not in stop_words and
                        len(token.text) > 2):
                        significant_words.append(token.lemma_.lower())
                
                # Stocker le mapping mot-clé → terme complet
                if significant_words:
                    # Mot-clé principal (premier mot significatif)
                    main_keyword = significant_words[0]
                    if main_keyword not in keywords[entity_type]:
                        keywords[entity_type][main_keyword] = []
                    keywords[entity_type][main_keyword].append(term)
                    
                    # Si c'est un terme composé, stocker aussi les combinaisons
                    if len(significant_words) > 1:
                        # Bigram principal
                        if len(significant_words) >= 2:
                            bigram = f"{significant_words[0]} {significant_words[1]}"
                            if bigram not in keywords[entity_type]:
                                keywords[entity_type][bigram] = []
                            keywords[entity_type][bigram].append(term)
        
        # Afficher les stats
        for entity_type in keywords:
            logger.info(f"  {entity_type}: {len(keywords[entity_type])} mots-cles extraits")
        
        return keywords
    
    def _create_matchers(self):
        """
        Crée les PhraseMatcher pour matching exact
        
        Returns:
            dict: Matchers par type d'entité
        """
        logger.info("Creation des matchers SpaCy pour matching exact...")
        
        matchers = {}
        
        for entity_type, terms in self.dictionaries.items():
            if not terms:
                continue
            
            matcher = PhraseMatcher(self.nlp.vocab, attr="LOWER")
            patterns = [self.nlp.make_doc(term) for term in terms]
            matcher.add(entity_type.upper(), patterns)
            matchers[entity_type] = matcher
            
            logger.info(f"  Matcher {entity_type}: {len(terms)} patterns")
        
        return matchers
    
    def _match_by_keyword(self, doc, entity_type):
        """
        Effectue un matching par mots-clés pour capturer les variations
        
        Args:
            doc: Document SpaCy
            entity_type: Type d'entité à chercher
            
        Returns:
            list: Matches trouvés avec (start, end, matched_term)
        """
        matches = []
        keywords_dict = self.keywords.get(entity_type, {})
        
        # Parcourir le document
        for i, token in enumerate(doc):
            if not token.is_alpha or len(token.text) < 3:
                continue
            
            lemme = token.lemma_.lower()
            
            # Vérifier si ce lemme est un mot-clé
            if lemme in keywords_dict:
                # Essayer de matcher un bigram d'abord
                if i < len(doc) - 1:
                    next_token = doc[i + 1]
                    if next_token.is_alpha:
                        bigram = f"{lemme} {next_token.lemma_.lower()}"
                        if bigram in keywords_dict:
                            # Match sur bigram
                            matched_terms = keywords_dict[bigram]
                            matches.append((i, i + 2, matched_terms[0]))
                            continue
                
                # Match sur unigram
                matched_terms = keywords_dict[lemme]
                matches.append((i, i + 1, matched_terms[0]))
        
        return matches
    
    def _is_valid_entity_context(self, span, entity_type):
        """
        Vérifie si le contexte d'une entité détectée est valide
        
        Args:
            span: Span SpaCy de l'entité
            entity_type: Type d'entité
            
        Returns:
            bool: True si le contexte est valide
        """
        # Vérifier que l'entité n'est pas un adjectif isolé
        if len(span) == 1 and span[0].pos_ == 'ADJ':
            return False
        
        # Vérifier que ce n'est pas une couleur (pour éviter "orangé" → "oranger")
        color_words = {'orange', 'orangé', 'jaune', 'vert', 'rouge', 'bleu', 'blanc', 'noir', 'gris', 'rose'}
        if span.text.lower() in color_words:
            return False
        
        # Pour les cultures, vérifier contexte spécial
        if entity_type == 'cultures':
            sentence = span.sent.text.lower()
            span_text = span.text.lower()
            
            # Exclure si fait partie d'un nom de ravageur/pathogène
            exclusion_patterns = [
                f"du {span_text}",  # "puceron vert du pêcher"
                f"de {span_text}",  # "altise de betterave"
                f"{span_text} et grosse",  # "petite et grosse" 
                f"petit et {span_text}",
                f"petite et {span_text}"
            ]
            
            for pattern in exclusion_patterns:
                if pattern in sentence:
                    # Vérifier que notre entité fait partie de ce pattern
                    start_pattern = sentence.find(pattern)
                    if start_pattern != -1:
                        end_pattern = start_pattern + len(pattern)
                        span_start_in_sent = span.start_char - span.sent.start_char
                        span_end_in_sent = span.end_char - span.sent.start_char
                        
                        # Si notre span est dans le pattern, exclure
                        if (span_start_in_sent >= start_pattern and 
                            span_end_in_sent <= end_pattern):
                            return False
        
        return True
    
    def _extract_entities_from_text(self, text):
        """
        Extrait toutes les entités d'un texte avec matching hybride et validation
        
        Args:
            text: Texte à analyser
            
        Returns:
            dict: Entités extraites avec contextes
        """
        doc = self.nlp(text)
        
        entities = {
            'cultures': defaultdict(lambda: {'count': 0, 'contexts': [], 'positions': []}),
            'pathogenes': defaultdict(lambda: {'count': 0, 'contexts': [], 'positions': []}),
            'ravageurs': defaultdict(lambda: {'count': 0, 'contexts': [], 'positions': []}),
            'symptomes': defaultdict(lambda: {'count': 0, 'contexts': [], 'positions': []})
        }
        
        # Stocker toutes les détections par type avec leur longueur
        all_detections = defaultdict(list)
        
        # Pour chaque type d'entité
        for entity_type in ['cultures', 'pathogenes', 'ravageurs', 'symptomes']:
            
            # 1. Matching exact avec PhraseMatcher
            if entity_type in self.matchers:
                exact_matches = self.matchers[entity_type](doc)
                
                for match_id, start, end in exact_matches:
                    span = doc[start:end]
                    entity_text = span.text.lower()
                    
                    # Validation contextuelle
                    if not self._is_valid_entity_context(span, entity_type):
                        continue
                    
                    all_detections[entity_type].append({
                        'start': start,
                        'end': end,
                        'entity_text': entity_text,
                        'span': span,
                        'length': end - start,
                        'source': 'exact'
                    })
            
            # 2. Matching par mots-clés pour variations
            keyword_matches = self._match_by_keyword(doc, entity_type)
            
            for start, end, matched_term in keyword_matches:
                span = doc[start:end]
                
                # Validation contextuelle
                if not self._is_valid_entity_context(span, entity_type):
                    continue
                
                all_detections[entity_type].append({
                    'start': start,
                    'end': end,
                    'entity_text': matched_term,
                    'span': span,
                    'length': end - start,
                    'source': 'keyword'
                })
        
        # Résoudre les chevauchements : priorité aux entités les plus longues
        for entity_type in ['cultures', 'pathogenes', 'ravageurs', 'symptomes']:
            detections = all_detections[entity_type]
            
            # Trier par longueur décroissante (les plus longues d'abord)
            detections.sort(key=lambda x: x['length'], reverse=True)
            
            seen_positions = set()
            
            for detection in detections:
                start = detection['start']
                end = detection['end']
                
                # Vérifier si cette position chevauche une position déjà vue
                overlaps = False
                for pos in range(start, end):
                    if pos in seen_positions:
                        overlaps = True
                        break
                
                if not overlaps:
                    # Marquer toutes les positions comme vues
                    for pos in range(start, end):
                        seen_positions.add(pos)
                    
                    # Ajouter l'entité
                    self._add_entity_occurrence(
                        entities[entity_type],
                        detection['entity_text'],
                        detection['span'],
                        entity_type
                    )
        
        return entities
    
    def _add_entity_occurrence(self, entity_dict, entity_text, span, entity_type):
        """
        Ajoute une occurrence d'entité avec son contexte
        
        Args:
            entity_dict: Dictionnaire des entités du type
            entity_text: Texte de l'entité
            span: Span SpaCy
            entity_type: Type d'entité
        """
        # Incrémenter le compteur
        entity_dict[entity_text]['count'] += 1
        entity_dict[entity_text]['positions'].append(span.start)
        
        # Ajouter le contexte (la phrase complète)
        sentence = span.sent.text.strip()
        
        # Top 5 pour cultures, Top 3 pour le reste
        max_contexts = 5 if entity_type == 'cultures' else 3
        if len(entity_dict[entity_text]['contexts']) < max_contexts:
            if sentence not in entity_dict[entity_text]['contexts']:
                entity_dict[entity_text]['contexts'].append(sentence)
    
    def _compute_cooccurrences(self, entities):
        """
        Calcule les co-occurrences entre types d'entités
        
        Args:
            entities: Dictionnaire des entités extraites
            
        Returns:
            dict: Matrices de co-occurrences
        """
        cooccurrences = {
            'cultures_x_pathogenes': [],
            'cultures_x_ravageurs': [],
            'ravageurs_x_symptomes': [],
            'cultures_x_symptomes': []
        }
        
        # Cultures × Pathogènes
        for culture in entities['cultures'].keys():
            for pathogene in entities['pathogenes'].keys():
                cooccurrences['cultures_x_pathogenes'].append({
                    'culture': culture,
                    'pathogene': pathogene,
                    'count_culture': entities['cultures'][culture]['count'],
                    'count_pathogene': entities['pathogenes'][pathogene]['count']
                })
        
        # Cultures × Ravageurs
        for culture in entities['cultures'].keys():
            for ravageur in entities['ravageurs'].keys():
                cooccurrences['cultures_x_ravageurs'].append({
                    'culture': culture,
                    'ravageur': ravageur,
                    'count_culture': entities['cultures'][culture]['count'],
                    'count_ravageur': entities['ravageurs'][ravageur]['count']
                })
        
        # Ravageurs × Symptômes
        for ravageur in entities['ravageurs'].keys():
            for symptome in entities['symptomes'].keys():
                cooccurrences['ravageurs_x_symptomes'].append({
                    'ravageur': ravageur,
                    'symptome': symptome,
                    'count_ravageur': entities['ravageurs'][ravageur]['count'],
                    'count_symptome': entities['symptomes'][symptome]['count']
                })
        
        # Cultures × Symptômes
        for culture in entities['cultures'].keys():
            for symptome in entities['symptomes'].keys():
                cooccurrences['cultures_x_symptomes'].append({
                    'culture': culture,
                    'symptome': symptome,
                    'count_culture': entities['cultures'][culture]['count'],
                    'count_symptome': entities['symptomes'][symptome]['count']
                })
        
        return cooccurrences
    
    def process_file(self, file_path, year):
        """
        Traite un seul fichier BSV pour extraire les entités
        
        Args:
            file_path: Chemin vers le fichier texte
            year: Année du fichier
            
        Returns:
            dict: Résultat de l'extraction ou None si erreur
        """
        try:
            # Lire le fichier
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            if not text.strip():
                logger.warning(f"Fichier vide: {file_path.name}")
                return None
            
            # Extraire les entités
            entities = self._extract_entities_from_text(text)
            
            # Nettoyer les positions (pas besoin dans le JSON final)
            for entity_type in entities:
                for entity_data in entities[entity_type].values():
                    del entity_data['positions']
            
            # Calculer les co-occurrences
            cooccurrences = self._compute_cooccurrences(entities)
            
            # Formater les résultats
            result = {
                "filename": file_path.name,
                "year": year,
                "entites_detectees": {},
                "statistiques": {
                    "nb_cultures": len(entities['cultures']),
                    "nb_pathogenes": len(entities['pathogenes']),
                    "nb_ravageurs": len(entities['ravageurs']),
                    "nb_symptomes": len(entities['symptomes']),
                    "total_entites": sum(len(entities[t]) for t in entities)
                },
                "co_occurrences": cooccurrences,
                "metadata": {
                    "date_extraction": time.strftime("%Y-%m-%d"),
                    "chemin_original": str(file_path)
                }
            }
            
            # Formater les entités détectées
            for entity_type in ['cultures', 'pathogenes', 'ravageurs', 'symptomes']:
                result['entites_detectees'][entity_type] = [
                    {
                        "entite": entity,
                        "occurrences": data['count'],
                        "contextes": data['contexts']
                    }
                    for entity, data in sorted(
                        entities[entity_type].items(),
                        key=lambda x: x[1]['count'],
                        reverse=True
                    )
                ]
            
            # Mise à jour des stats globales
            for entity_type in ['cultures', 'pathogenes', 'ravageurs', 'symptomes']:
                self.global_stats['total_entities'][entity_type] += len(entities[entity_type])
                self.global_stats['unique_entities'][entity_type].update(entities[entity_type].keys())
                
                # Compteur global
                for entity, data in entities[entity_type].items():
                    self.global_entity_counter[entity_type][entity] += data['count']
            
            return result
            
        except Exception as e:
            logger.error(f"Erreur lors du traitement de {file_path.name}: {e}")
            logger.exception(e)
            self.global_stats['errors'].append({
                'file': str(file_path),
                'error': str(e)
            })
            return None
    
    def process_all_files(self):
        """
        Traite tous les fichiers BSV du dossier source
        """
        logger.info("Debut de l'extraction d'entites sur les fichiers BSV")
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
        Génère un rapport de synthèse de l'extraction d'entités
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
                "temps_traitement_secondes": round(self.global_stats['processing_time'], 2)
            },
            "entites_par_type": {},
            "top_entites_par_type": {},
            "erreurs": self.global_stats['errors']
        }
        
        # Statistiques par type d'entité
        for entity_type in ['cultures', 'pathogenes', 'ravageurs', 'symptomes']:
            report['entites_par_type'][entity_type] = {
                'total_occurrences': self.global_stats['total_entities'][entity_type],
                'entites_uniques': len(self.global_stats['unique_entities'][entity_type])
            }
            
            # Top 20 entités par type
            report['top_entites_par_type'][entity_type] = [
                {"entite": entity, "frequency": freq}
                for entity, freq in self.global_entity_counter[entity_type].most_common(20)
            ]
        
        # Sauvegarder le rapport
        report_path = self.output_dir / "entity_extraction_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Rapport sauvegarde: {report_path}")
        
        # Afficher le résumé
        logger.info("=" * 80)
        logger.info("EXTRACTION D'ENTITES TERMINEE")
        logger.info("=" * 80)
        logger.info(f"Fichiers traites: {report['statistiques']['fichiers_traites']}/{report['statistiques']['total_fichiers']}")
        logger.info(f"Temps de traitement: {report['statistiques']['temps_traitement_secondes']}s")
        logger.info("")
        
        for entity_type in ['cultures', 'pathogenes', 'ravageurs', 'symptomes']:
            stats = report['entites_par_type'][entity_type]
            logger.info(f"{entity_type.upper()}: {stats['entites_uniques']} entites uniques")
            
            # Top 5
            top_entities = report['top_entites_par_type'][entity_type][:5]
            for i, item in enumerate(top_entities, 1):
                logger.info(f"  {i}. {item['entite']}: {item['frequency']} occurrences")
        
        logger.info("=" * 80)


def main():
    """
    Fonction principale avec arguments en ligne de commande
    """
    parser = argparse.ArgumentParser(
        description="Extraction d'entites nommees des fichiers BSV avec matching intelligent"
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
    logger.info("Demarrage de l'extraction d'entites (version matching intelligent)")
    logger.info("Plant Health NLP Analysis - Polytech Dijon")
    
    try:
        extractor = EntityExtractor(
            force_reprocess=args.force,
            target_year=args.year
        )
        extractor.process_all_files()
        
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