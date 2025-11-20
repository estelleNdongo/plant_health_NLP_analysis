import re
import os
import sys
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config_loader import ConfigLoader
from utils.logger import setup_logging, get_logger

# Initialiser le logger
logger = get_logger(__name__)

class BSVCleaner:
    """
    Classe pour nettoyer les fichiers BSV texte et les organiser dans clean_txt
    """
    
    def __init__(self):
        # Charger la config
        self.config_loader = ConfigLoader("config.yaml")
        
        # Chemins depuis la config
        self.processed_base_dir = self.config_loader.get_path(self.config_loader.config["data"]["processed_dir"])
        
        # Chemins source et destination
        self.source_dir = os.path.join(self.processed_base_dir, "txt", "bourgogne_franche_comte")
        self.dest_dir = os.path.join(self.processed_base_dir, "clean_txt", "bourgogne_franche_comte")
        
        # Compiler les regex pour la performance
        self._compiler_regex()

    def _compiler_regex(self):
        """Compile toutes les expressions régulières"""
        self.regex_patterns = {
            'pages_isoles': re.compile(r'^\s*\d+\s*$', re.MULTILINE),
            'headers_repetitifs': re.compile(r'(?<=\n)(Grandes cultures n° \d+ du \d+ \d+ \d+|N°\d+ du \d{2}/\d{2}/\d{4})\s*$\n', re.MULTILINE),
            'mots_coupes': re.compile(r'(\w+)-\n\s*(\w+)'),
            'puces': re.compile(r'^[•]\s*', re.MULTILINE),
            'espaces_multiples': re.compile(r'[ ]{2,}'),
            'lignes_coupees': re.compile(r'([a-zàâäéèêëïîôöùûüÿç,;])\n([a-zàâäéèêëïîôöùûüÿç])'),
            'lignes_espaces': re.compile(r'^[ \t]+\n', re.MULTILINE),
            'lignes_vides_excessives': re.compile(r'\n{3,}')
        }

    def nettoyer_contenu(self, contenu):
        """
        Nettoie un contenu de BSV en préservant la structure
        
        Args:
            contenu (str): Contenu brut du BSV
            
        Returns:
            str: Contenu nettoyé
        """
        contenu_avant = contenu
        
        # Appliquer toutes les étapes de nettoyage
        etapes_nettoyage = [
            self._supprimer_pages_isoles,
            self._supprimer_headers_repetitifs,
            self._reformer_mots_coupes,
            self._uniformiser_puces,
            self._supprimer_espaces_multiples,
            self._fusionner_lignes_coupees,
            self._optimiser_lignes_vides,
            self._nettoyer_bords
        ]
        
        for etape in etapes_nettoyage:
            contenu = etape(contenu)
        
        return contenu

    def _supprimer_pages_isoles(self, contenu):
        """Supprime les numéros de pages isolés"""
        return self.regex_patterns['pages_isoles'].sub('', contenu)

    def _supprimer_headers_repetitifs(self, contenu):
        """Supprime les headers répétitifs"""
        return self.regex_patterns['headers_repetitifs'].sub('', contenu)

    def _reformer_mots_coupes(self, contenu):
        """Reforme les mots coupés par des tirets"""
        return self.regex_patterns['mots_coupes'].sub(r'\1\2', contenu)

    def _uniformiser_puces(self, contenu):
        """Uniformise les listes à puces"""
        return self.regex_patterns['puces'].sub('- ', contenu)

    def _supprimer_espaces_multiples(self, contenu):
        """Supprime les espaces multiples dans les lignes"""
        return self.regex_patterns['espaces_multiples'].sub(' ', contenu)

    def _fusionner_lignes_coupees(self, contenu):
        """Fusionne uniquement les lignes clairement coupées"""
        return self.regex_patterns['lignes_coupees'].sub(r'\1 \2', contenu)

    def _optimiser_lignes_vides(self, contenu):
        """Optimise les lignes vides"""
        contenu = self.regex_patterns['lignes_espaces'].sub('\n', contenu)
        return self.regex_patterns['lignes_vides_excessives'].sub('\n\n', contenu)

    def _nettoyer_bords(self, contenu):
        """Nettoie les bords du contenu"""
        return contenu.strip()

    def nettoyer_fichier(self, chemin_entree, chemin_sortie):
        """
        Nettoie un fichier BSV et le sauvegarde
        
        Args:
            chemin_entree (str): Chemin vers le fichier d'entrée
            chemin_sortie (str): Chemin vers le fichier de sortie
            
        Returns:
            bool: True si succès, False sinon
        """
        try:
            with open(chemin_entree, 'r', encoding='utf-8') as f:
                contenu_original = f.read()
            
            contenu_nettoye = self.nettoyer_contenu(contenu_original)
            
            os.makedirs(os.path.dirname(chemin_sortie), exist_ok=True)
            
            with open(chemin_sortie, 'w', encoding='utf-8') as f:
                f.write(contenu_nettoye)
            
            return True
            
        except Exception as e:
            logger.error(f"Erreur lors du nettoyage de {chemin_entree}: {e}")
            return False

    def obtenir_statistiques(self, contenu_original, contenu_nettoye):
        """
        Retourne les statistiques de nettoyage
        
        Returns:
            dict: Statistiques de nettoyage
        """
        return {
            'lignes_original': contenu_original.count('\n'),
            'lignes_nettoye': contenu_nettoye.count('\n'),
            'caracteres_original': len(contenu_original),
            'caracteres_nettoye': len(contenu_nettoye),
            'reduction_pourcentage': ((len(contenu_original) - len(contenu_nettoye)) / len(contenu_original) * 100) if contenu_original else 0
        }

    def nettoyer_tous_fichiers(self):
        """
        Nettoie tous les fichiers .txt du dossier source et les sauvegarde dans clean_txt
        en conservant la même structure
        """
        logger.info("Debut du nettoyage des fichiers BSV")
        
        if not os.path.exists(self.source_dir):
            logger.error(f"Dossier source non trouve: {self.source_dir}")
            return 0, 0
        
        total_files = 0
        success_files = 0
        stats_totales = {
            'lignes_original': 0,
            'lignes_nettoye': 0,
            'caracteres_original': 0,
            'caracteres_nettoye': 0
        }
        
        for root, dirs, files in os.walk(self.source_dir):
            for file in files:
                if file.endswith('.txt'):
                    total_files += 1
                    chemin_entree = os.path.join(root, file)
                    
                    # Construire le chemin de sortie en conservant la structure
                    relative_path = os.path.relpath(root, self.source_dir)
                    chemin_sortie = os.path.join(self.dest_dir, relative_path, file)
                    
                    if self.nettoyer_fichier(chemin_entree, chemin_sortie):
                        success_files += 1
                        
                        # Calculer les statistiques pour ce fichier
                        with open(chemin_entree, 'r', encoding='utf-8') as f:
                            contenu_original = f.read()
                        with open(chemin_sortie, 'r', encoding='utf-8') as f:
                            contenu_nettoye = f.read()
                        
                        stats = self.obtenir_statistiques(contenu_original, contenu_nettoye)
                        
                        # Accumuler les statistiques totales
                        for key in stats_totales:
                            stats_totales[key] += stats[key]
                        
                        logger.info(f"Fichier nettoye: {chemin_sortie} "
                                  f"({stats['reduction_pourcentage']:.1f}% de reduction)")
                    else:
                        logger.error(f"Echec du nettoyage: {chemin_entree}")
        
        # Afficher les statistiques globales
        if total_files > 0:
            reduction_moyenne = ((stats_totales['caracteres_original'] - stats_totales['caracteres_nettoye']) / 
                               stats_totales['caracteres_original'] * 100)
            
            logger.info(f"Nettoyage termine: {success_files}/{total_files} fichiers traites avec succes")
            logger.info(f"Statistiques globales:")
            logger.info(f"  Lignes: {stats_totales['lignes_original']} -> {stats_totales['lignes_nettoye']}")
            logger.info(f"  Caracteres: {stats_totales['caracteres_original']} -> {stats_totales['caracteres_nettoye']}")
            logger.info(f"  Reduction moyenne: {reduction_moyenne:.1f}%")
        
        return success_files, total_files


def main():
    """Fonction principale"""
    setup_logging()
    logger.info("Demarrage du nettoyage des fichiers BSV")
    logger.info("Plant Health NLP Analysis - Polytech Dijon")

    try:
        cleaner = BSVCleaner()
        success, total = cleaner.nettoyer_tous_fichiers()
        
        if success == total:
            logger.info("Nettoyage termine avec succes!")
            return 0
        else:
            logger.warning(f"Nettoyage partiel: {success}/{total} fichiers")
            return 1

    except KeyboardInterrupt:
        logger.warning("Interruption par l'utilisateur (Ctrl+C)")
        return 1

    except Exception as e:
        logger.error("Erreur fatale lors du nettoyage")
        logger.exception(e)
        return 1


if __name__ == "__main__":
    sys.exit(main())