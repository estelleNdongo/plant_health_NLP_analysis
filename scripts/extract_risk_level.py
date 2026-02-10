import spacy
from spacy.matcher import Matcher
import json
import re
from pathlib import Path
from collections import defaultdict, Counter
import sys
import os
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config_loader import ConfigLoader
from utils.logger import setup_logging, get_logger

logger = get_logger(__name__)


class RiskExtractorNLP:
    """
    Extraction de niveaux de risque par NLP pur (syntaxe + embeddings)
    Sans dictionnaire de keywords !
    """
    
    def __init__(self):
        """Initialise l'extracteur NLP"""
        self.config_loader = ConfigLoader("config.yaml")
        self.base_dir = Path(self.config_loader.base_dir)
        
        # Chemins
        self.processed_dir = self.base_dir / self.config_loader.config["data"]["processed_dir"]
        self.results_dir = self.base_dir / self.config_loader.config["data"]["results_dir"]
        self.source_dir = self.processed_dir / "clean_txt" / "bourgogne_franche_comte"
        self.output_dir = self.results_dir / "risk_analysis_nlp"
        self.entities_dir = self.results_dir / "entities"
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Charger SpaCy avec embeddings
        logger.info("Chargement du modele SpaCy avec embeddings...")
        try:
            self.nlp = spacy.load("fr_core_news_lg")
            logger.info("Modele fr_core_news_lg charge avec succes")
        except:
            logger.error("Modele fr_core_news_lg non trouve")
            sys.exit(1)
        
        # Créer les matchers syntaxiques
        self.matcher = self._create_syntax_matchers()
        
        # Mots de référence pour classification par similarité
        self.reference_words = self._create_reference_embeddings()
        
        # Charger les entités connues
        self.known_entities = self._load_known_entities()
        
        # Statistiques
        self.stats = {
            'total_files': 0,
            'processed_files': 0,
            'total_risk_mentions': 0,
            'risk_by_level': Counter(),
            'risk_by_entity': defaultdict(Counter)
        }
        
        logger.info("Extracteur NLP initialise avec succes")
    
    def _create_reference_embeddings(self):
        """
        Crée les embeddings de référence pour chaque niveau de risque
        
        Returns:
            dict: Embeddings moyens par niveau
        """
        logger.info("Creation des embeddings de reference...")
        
        reference_sets = {
            'très faible': ['nul', 'inexistant', 'négligeable', 'aucun'],
            'faible': ['faible', 'bas', 'limité', 'réduit', 'léger'],
            'modéré': ['modéré', 'moyen', 'intermédiaire', 'notable'],
            'élevé': ['élevé', 'fort', 'important', 'significatif', 'conséquent'],
            'très élevé': ['critique', 'alarmant', 'généralisé', 'majeur', 'urgent']
        }
        
        reference_embeddings = {}
        
        for level, words in reference_sets.items():
            # Calculer l'embedding moyen des mots de référence
            vectors = []
            for word in words:
                doc = self.nlp(word)
                if doc[0].has_vector:
                    vectors.append(doc[0].vector)
            
            if vectors:
                # Moyenne des vecteurs
                reference_embeddings[level] = np.mean(vectors, axis=0)
                logger.info(f"  {level}: {len(vectors)} mots de reference")
        
        return reference_embeddings
    
    def _create_syntax_matchers(self):
        """
        Crée les matchers pour patterns syntaxiques
        
        Returns:
            Matcher: Matcher configuré
        """
        logger.info("Creation des matchers syntaxiques...")
        
        matcher = Matcher(self.nlp.vocab)
        
        # Pattern 1: "risque [ENTITY] est/devient/reste ADJ"
        # Ex: "Le risque altise est élevé", "Le risque devient important"
        pattern1 = [
            {"LOWER": "risque"},
            {"POS": {"IN": ["NOUN", "PROPN"]}, "OP": "*"},  # Entité optionnelle
            {"LEMMA": {"IN": ["être", "devenir", "rester"]}},  # Verbes acceptés
            {"POS": "ADJ"}  # Le niveau !
        ]
        matcher.add("RISK_BE_ADJ", [pattern1])
        
        # Pattern 2: "risque ADJ"
        # Ex: "risque élevé", "risque faible"
        pattern2 = [
            {"LOWER": "risque"},
            {"POS": "ADJ", "OP": "+"}  # Un ou plusieurs adjectifs
        ]
        matcher.add("RISK_ADJ", [pattern2])
        
        # Pattern 3: "pression ADJ"
        # Ex: "pression forte", "pression limitée"
        pattern3 = [
            {"LOWER": "pression"},
            {"POS": {"IN": ["NOUN", "PROPN"]}, "OP": "*"},
            {"POS": "ADJ"}
        ]
        matcher.add("PRESSION_ADJ", [pattern3])
        
        # Pattern 4: "ADJ présence/pression/observation"
        # Ex: "forte présence", "faible observation"
        pattern4 = [
            {"POS": "ADJ"},
            {"LOWER": {"IN": ["présence", "pression", "observation"]}}
        ]
        matcher.add("ADJ_PRESENCE", [pattern4])
        
        # Pattern 5: "seuil [ADJ] [dépassé/atteint]"
        # Ex: "seuil dépassé", "seuil largement dépassé"
        pattern5 = [
            {"LOWER": "seuil"},
            {"POS": "ADV", "OP": "*"},  # Adverbe optionnel
            {"LEMMA": {"IN": ["dépasser", "atteindre", "franchir"]}}
        ]
        matcher.add("SEUIL_DEPASSE", [pattern5])
        
        # Pattern 6: Négations "pas de risque", "aucun risque"
        pattern6 = [
            {"LOWER": {"IN": ["pas", "aucun", "aucune"]}},
            {"LOWER": {"IN": ["de", "d"]}, "OP": "?"},
            {"LOWER": {"IN": ["risque", "pression", "dépassement"]}}
        ]
        matcher.add("NEGATION_RISK", [pattern6])
        
        logger.info(f"  {len(matcher)} patterns syntaxiques crees")
        
        return matcher
    
    def _load_known_entities(self):
        """Charge les entités connues depuis les rapports précédents et les listes"""
        entities = {
            'ravageurs': set(),
            'pathogenes': set(),
            'cultures': set()
        }
        
        # Charger depuis le rapport d'extraction
        report_path = self.entities_dir / "entity_extraction_report.json"
        if report_path.exists():
            try:
                with open(report_path, 'r', encoding='utf-8') as f:
                    report = json.load(f)
                    
                for entity_type in ['ravageurs', 'pathogenes', 'cultures']:
                    if entity_type in report.get('top_entites_par_type', {}):
                        for item in report['top_entites_par_type'][entity_type]:
                            entities[entity_type].add(item['entite'].lower())
                
                logger.info(f"Entites du rapport: {len(entities['ravageurs'])} ravageurs, "
                          f"{len(entities['pathogenes'])} pathogenes, {len(entities['cultures'])} cultures")
            except Exception as e:
                logger.warning(f"Erreur chargement rapport: {e}")
        
        # Charger les pathogènes depuis list/pathogenes.txt (enrichi)
        pathogenes_path = self.base_dir / "list" / "pathogenes.txt"
        if pathogenes_path.exists():
            try:
                count_before = len(entities['pathogenes'])
                with open(pathogenes_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        pathogene = line.strip().lower()
                        if pathogene:
                            entities['pathogenes'].add(pathogene)
                count_after = len(entities['pathogenes'])
                logger.info(f"Pathogenes charges depuis list/pathogenes.txt: +{count_after - count_before} nouveaux")
            except Exception as e:
                logger.warning(f"Erreur chargement pathogenes.txt: {e}")
        
        logger.info(f"TOTAL: {len(entities['ravageurs'])} ravageurs, "
                   f"{len(entities['pathogenes'])} pathogenes, {len(entities['cultures'])} cultures")
        
        return entities
    
    def _classify_adjective_by_similarity(self, adj_text):
        """
        Classifie un adjectif par similarité avec les embeddings de référence
        
        Args:
            adj_text: Texte de l'adjectif
            
        Returns:
            tuple: (niveau, score, confiance)
        """
        adj_doc = self.nlp(adj_text)
        
        if not adj_doc[0].has_vector:
            return None, None, 0.0
        
        adj_vector = adj_doc[0].vector
        
        # Calculer la similarité avec chaque niveau
        similarities = {}
        for level, ref_vector in self.reference_words.items():
            # Similarité cosinus
            sim = np.dot(adj_vector, ref_vector) / (
                np.linalg.norm(adj_vector) * np.linalg.norm(ref_vector)
            )
            similarities[level] = sim
        
        # Trouver le niveau avec la plus haute similarité
        best_level = max(similarities, key=similarities.get)
        confidence = similarities[best_level]
        
        # Mapping niveau → score
        score_map = {
            'très faible': 0,
            'faible': 1,
            'modéré': 2,
            'élevé': 3,
            'très élevé': 4
        }
        
        return best_level, score_map[best_level], confidence
    
    def _handle_negation(self, span):
        """
        Détecte si le span contient une négation
        
        Args:
            span: Span SpaCy
            
        Returns:
            bool: True si négation détectée
        """
        # Chercher "pas", "aucun", "ne...pas" avant le span
        for i in range(max(0, span.start - 5), span.start):
            token = span.doc[i]
            if token.text.lower() in ['pas', 'aucun', 'aucune', 'ne', 'non', 'sans']:
                return True
        
        # Chercher dans le span lui-même
        for token in span:
            if token.text.lower() in ['pas', 'aucun', 'aucune', 'ne', 'non', 'sans']:
                return True
        
        return False
    
    def _extract_adjectives_from_span(self, span):
        """
        Extrait tous les adjectifs d'un span
        
        Args:
            span: Span SpaCy
            
        Returns:
            list: Liste d'adjectifs
        """
        adjectives = []
        for token in span:
            if token.pos_ == "ADJ":
                adjectives.append(token)
        return adjectives
    
    def _detect_pathogen_by_pattern(self, text_lower):
        """
        Détecte automatiquement les pathogènes par patterns même s'ils ne sont pas listés
        
        Args:
            text_lower: Texte en minuscules
            
        Returns:
            list: Pathogènes détectés
        """
        import re
        
        detected = []
        
        # Pattern 1: Maladies en -ose, -iose
        diseases = re.findall(r'\b\w+(?:i?ose)\b', text_lower)
        for disease in diseases:
            if len(disease) > 5:  # Éviter "chose", "rose", etc.
                detected.append(disease)
        
        # Pattern 2: Noms latins (Genre espèce)
        # Ex: "Fusarium graminearum", "Phoma lingam"
        latin_names = re.findall(r'\b[A-Z][a-z]+\s+[a-z]+(?:\s+f\.\s+sp\.\s+[a-z]+)?\b', text_lower)
        detected.extend(latin_names)
        
        # Pattern 3: Mots-clés spécifiques
        keywords = ['mildiou', 'oïdium', 'oidium', 'rouille', 'phomopsis', 
                   'sclérotinia', 'sclerotinia', 'botrytis', 'pythium',
                   'blanc', 'tavelure', 'moniliose', 'chancre']
        
        for keyword in keywords:
            if keyword in text_lower:
                detected.append(keyword)
        
        return list(set(detected))  # Dédupliquer
    
    def _find_nearby_entities(self, sentence, window=40):  # Augmenté à 40
        """
        Trouve les entités dans la phrase ou à proximité
        + Détection automatique de pathogènes par patterns
        
        Args:
            sentence: Phrase SpaCy
            window: Fenêtre de recherche (tokens)
            
        Returns:
            dict: Entités trouvées par type
        """
        found = {
            'ravageurs': [],
            'pathogenes': [],
            'cultures': []
        }
        
        # Chercher dans une fenêtre autour de la phrase
        start_idx = max(0, sentence.start - window)
        end_idx = min(len(sentence.doc), sentence.end + window)
        
        search_text = sentence.doc[start_idx:end_idx].text.lower()
        
        for entity_type in ['ravageurs', 'pathogenes', 'cultures']:
            for entity in self.known_entities[entity_type]:
                # Match exact
                if entity in search_text:
                    found[entity_type].append(entity)
                    continue
                
                # Match partiel : chercher les mots-clés de l'entité
                # Ex: "puceron vert du pêcher" → chercher "puceron"
                entity_words = entity.split()
                for word in entity_words:
                    if len(word) > 4 and word in search_text:  # Mots significatifs
                        found[entity_type].append(entity)
                        break
        
        # Détection automatique de pathogènes par patterns
        auto_pathogenes = self._detect_pathogen_by_pattern(search_text)
        for pathogene in auto_pathogenes:
            if pathogene not in found['pathogenes']:
                found['pathogenes'].append(pathogene)
        
        return found
    
    def _extract_month_from_filename(self, filename):
        """Extrait le mois du nom de fichier"""
        match = re.search(r'du_\d{2}_(\d{2})_\d{2,4}', filename)
        if match:
            try:
                month = int(match.group(1))
                if 1 <= month <= 12:
                    return month
            except:
                pass
        return None
    
    def extract_risks_from_text(self, text, filename):
        """
        Extrait les niveaux de risque par NLP pur
        
        Args:
            text: Texte du BSV
            filename: Nom du fichier
            
        Returns:
            list: Risques extraits
        """
        doc = self.nlp(text)
        risks = []
        month = self._extract_month_from_filename(filename)
        
        # Appliquer les matchers
        matches = self.matcher(doc)
        
        for match_id, start, end in matches:
            span = doc[start:end]
            match_label = self.nlp.vocab.strings[match_id]
            
            # Vérifier négation
            has_negation = self._handle_negation(span)
            
            # Extraire les adjectifs du match
            adjectives = self._extract_adjectives_from_span(span)
            
            if not adjectives and match_label == "SEUIL_DEPASSE":
                # "seuil dépassé" = élevé
                level = "élevé"
                score = 3
                confidence = 0.95
            elif not adjectives and match_label == "NEGATION_RISK":
                # "pas de risque" = très faible
                level = "très faible"
                score = 0
                confidence = 0.95
            elif adjectives:
                # Classifier le premier adjectif par similarité
                adj_text = adjectives[0].text
                level, score, confidence = self._classify_adjective_by_similarity(adj_text)
                
                if level is None:
                    continue
                
                # Inverser si négation
                if has_negation:
                    # Inverser le score
                    score = 4 - score
                    level_map = {0: 'très faible', 1: 'faible', 2: 'modéré', 3: 'élevé', 4: 'très élevé'}
                    level = level_map[score]
                    logger.debug(f"Negation detectee: inversion {adj_text} → {level}")
            else:
                continue
            
            # Seuil de confiance minimum
            if confidence < 0.3:
                continue
            
            # Chercher les entités associées
            sentence = span.sent
            entities = self._find_nearby_entities(sentence)
            
            # Construire l'entrée de risque
            risk_entry = {
                'niveau': level,
                'score': int(score),  # Convertir en int Python natif
                'confiance': float(round(confidence, 2)),  # Convertir en float Python natif
                'ravageurs': entities['ravageurs'],
                'pathogenes': entities['pathogenes'],
                'cultures': entities['cultures'],
                'contexte': sentence.text.strip(),
                'mois': month,
                'adjectif_detecte': adjectives[0].text if adjectives else None,
                'pattern_matche': match_label
            }
            
            risks.append(risk_entry)
            
            # Stats
            self.stats['total_risk_mentions'] += 1
            self.stats['risk_by_level'][level] += 1
            
            for ravageur in entities['ravageurs']:
                self.stats['risk_by_entity'][ravageur][level] += 1
            for pathogene in entities['pathogenes']:
                self.stats['risk_by_entity'][pathogene][level] += 1
        
        return risks
    
    def process_file(self, file_path, year):
        """
        Traite un fichier BSV
        
        Args:
            file_path: Chemin du fichier
            year: Année
            
        Returns:
            dict: Résultats
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            if not text.strip():
                return None
            
            risks = self.extract_risks_from_text(text, file_path.name)
            
            # Statistiques du fichier
            risk_summary = {
                'total_mentions': len(risks),
                'niveaux_detectes': Counter([r['niveau'] for r in risks]),
                'score_moyen': float(sum(r['score'] for r in risks) / len(risks)) if risks else 0.0,
                'confiance_moyenne': float(sum(r['confiance'] for r in risks) / len(risks)) if risks else 0.0,
                'entites_a_risque': list(set(
                    [e for r in risks for e in r['ravageurs']] +
                    [e for r in risks for e in r['pathogenes']]
                ))
            }
            
            result = {
                'filename': file_path.name,
                'year': year,
                'risques_detectes': risks,
                'statistiques': risk_summary,
                'methode': 'NLP pur (syntaxe + embeddings)'
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Erreur traitement {file_path.name}: {e}")
            return None
    
    def process_all_files(self):
        """Traite tous les fichiers BSV"""
        logger.info("Debut de l'extraction NLP pure")
        
        if not self.source_dir.exists():
            logger.error(f"Dossier source inexistant: {self.source_dir}")
            return
        
        all_results = []
        
        for year_dir in sorted(self.source_dir.iterdir()):
            if not year_dir.is_dir():
                continue
            
            year = year_dir.name
            logger.info(f"Traitement de l'annee {year}")
            
            output_year_dir = self.output_dir / year
            output_year_dir.mkdir(parents=True, exist_ok=True)
            
            txt_files = list(year_dir.glob("*.txt"))
            self.stats['total_files'] += len(txt_files)
            
            for file_path in txt_files:
                logger.info(f"Analyse: {file_path.name}")
                result = self.process_file(file_path, year)
                
                if result:
                    output_path = output_year_dir / f"{file_path.stem}_risks_nlp.json"
                    with open(output_path, 'w', encoding='utf-8') as f:
                        json.dump(result, f, ensure_ascii=False, indent=2)
                    
                    all_results.append(result)
                    self.stats['processed_files'] += 1
        
        self._generate_report(all_results)
    
    def _generate_report(self, all_results):
        """Génère un rapport de synthèse"""
        logger.info("Generation du rapport de synthese NLP")
        
        risk_by_entity_summary = {}
        for entity, levels in self.stats['risk_by_entity'].items():
            total = sum(levels.values())
            if total >= 3:
                risk_by_entity_summary[entity] = {
                    'total_mentions': total,
                    'repartition': dict(levels),
                    'score_moyen': sum(
                        {'très faible': 0, 'faible': 1, 'modéré': 2, 'élevé': 3, 'très élevé': 4}[level] * count 
                        for level, count in levels.items()
                    ) / total
                }
        
        top_risky_entities = sorted(
            risk_by_entity_summary.items(),
            key=lambda x: x[1]['score_moyen'],
            reverse=True
        )[:20]
        
        report = {
            'date_generation': self._get_timestamp(),
            'methode': 'NLP pur (syntaxe + embeddings)',
            'statistiques_globales': {
                'fichiers_analyses': self.stats['processed_files'],
                'mentions_risque_total': self.stats['total_risk_mentions'],
                'repartition_niveaux': dict(self.stats['risk_by_level'])
            },
            'top_entites_a_risque': [
                {
                    'entite': entity,
                    'score_moyen': round(data['score_moyen'], 2),
                    'mentions_total': data['total_mentions'],
                    'repartition': data['repartition']
                }
                for entity, data in top_risky_entities
            ],
            'analyse_par_niveau': {
                level: {
                    'occurrences': count,
                    'pourcentage': round(count / self.stats['total_risk_mentions'] * 100, 1) 
                                  if self.stats['total_risk_mentions'] > 0 else 0
                }
                for level, count in self.stats['risk_by_level'].items()
            }
        }
        
        report_path = self.output_dir / "risk_analysis_nlp_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Rapport sauvegarde: {report_path}")
        self._print_summary(report)
    
    def _print_summary(self, report):
        """Affiche un résumé"""
        logger.info("=" * 80)
        logger.info("EXTRACTION NLP PURE - RÉSUMÉ")
        logger.info("=" * 80)
        logger.info(f"Methode: {report['methode']}")
        logger.info(f"Fichiers analyses: {report['statistiques_globales']['fichiers_analyses']}")
        logger.info(f"Mentions de risque: {report['statistiques_globales']['mentions_risque_total']}")
        logger.info("")
        logger.info("Répartition par niveau:")
        for level, data in report['analyse_par_niveau'].items():
            logger.info(f"  {level}: {data['occurrences']} ({data['pourcentage']}%)")
        logger.info("")
        logger.info("Top 10 entités à risque élevé:")
        for i, item in enumerate(report['top_entites_a_risque'][:10], 1):
            logger.info(f"  {i}. {item['entite']}: score moyen {item['score_moyen']}/4")
        logger.info("=" * 80)
    
    def _get_timestamp(self):
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def main():
    """Fonction principale"""
    setup_logging()
    logger.info("Demarrage de l'extraction NLP pure (syntaxe + embeddings)")
    logger.info("Plant Health NLP Analysis - Polytech Dijon")
    
    try:
        extractor = RiskExtractorNLP()
        extractor.process_all_files()
        
        logger.info("Extraction NLP pure terminee avec succes!")
        return 0
        
    except KeyboardInterrupt:
        logger.warning("Interruption par l'utilisateur (Ctrl+C)")
        return 1
        
    except Exception as e:
        logger.error("Erreur fatale")
        logger.exception(e)
        return 1


if __name__ == "__main__":
    sys.exit(main())