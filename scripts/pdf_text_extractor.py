import pymupdf
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config_loader import ConfigLoader
from utils.logger import setup_logging, get_logger

# Initialiser le logger
logger = get_logger(__name__)

class PDFTextExtractor:
    def __init__(self):
        # Charger la config avec ton ConfigLoader
        self.config_loader = ConfigLoader("config.yaml")
        
        # Chemins depuis la config
        self.raw_base_dir = self.config_loader.get_path(self.config_loader.config["data"]["raw_dir"])
        self.processed_base_dir = self.config_loader.get_path(self.config_loader.config["data"]["processed_dir"])
        self.bourgogne_raw_dir = self.config_loader.config['scraping']['regions']['bourgogne_franche_comte']['output_dir_pase_path']
        
        # Chemin complet 
        self.raw_full_path = self.config_loader.get_path(self.bourgogne_raw_dir)

    def extract_text_from_pdf(self, pdf_path, output_path):
        """Extrait le texte d'un PDF et le sauvegarde dans un fichier texte"""
        try:
            with pymupdf.open(pdf_path) as doc:
                with open(output_path, "w", encoding="utf8") as out:
                    for page in doc:
                        text = page.get_text()
                        out.write(text)
                        out.write("\n")
                        out.write("\n")
            return True
        except Exception as e:
            logger.error(f"Erreur lors de l'extraction de {pdf_path}: {e}")
            return False

    def process_all_pdfs(self):
        """Traite tous les PDF du dossier et extrait le texte"""
        logger.info("Debut de l'extraction texte des PDF")
        
        total_files = 0
        success_files = 0
        
        for root, dirs, files in os.walk(self.raw_full_path):
            for file in files:
                if file.endswith('.pdf'):
                    total_files += 1
                    pdf_path = os.path.join(root, file)

                    relative_path = os.path.relpath(root, self.raw_full_path)
                    
                    output_dir = os.path.join(self.processed_base_dir, 'bourgogne_franche_comte', relative_path)
                    os.makedirs(output_dir, exist_ok=True)
                    txt_filename = os.path.splitext(file)[0] + '.txt'
                    output_path = os.path.join(output_dir, txt_filename) 
                    
                    if self.extract_text_from_pdf(pdf_path, output_path):
                        success_files += 1
                        logger.info(f"Texte extrait: {output_path}")
                    else:
                        logger.error(f"Echec extraction: {pdf_path}")

        logger.info(f"Extraction terminee: {success_files}/{total_files} fichiers traites avec succes")
        return success_files, total_files


def main():
    """Fonction principale avec gestion d'erreurs"""
    setup_logging()
    logger.info("Demarrage de l'extraction texte PDF")
    logger.info("Plant Health NLP Analysis - Polytech Dijon")

    try:
        extractor = PDFTextExtractor()
        success, total = extractor.process_all_pdfs()
        
        if success == total:
            logger.info("Extraction terminee avec succes!")
            return 0
        else:
            logger.warning(f"Extraction partielle: {success}/{total} fichiers")
            return 1

    except KeyboardInterrupt:
        logger.warning("Interruption par l'utilisateur (Ctrl+C)")
        return 1

    except Exception as e:
        logger.error("Erreur fatale lors de l'extraction PDF")
        logger.exception(e)
        return 1


if __name__ == "__main__":
    sys.exit(main())