import os
import sys

import pdfplumber
from pathlib import Path

from utils.config_loader import ConfigLoader
from utils.file_utils import logger
from utils.logger import setup_logging


class TextExtractor:

    def __init__(self):
        self.cfg = ConfigLoader().config
        self.base_directory_path = ConfigLoader().base_dir
        logger.info("Initialisation de l'extracteur de texte")
        logger.debug(f"Répertoire de base: {self.base_directory_path}")


    def extract_text_pdfplumber(self, region="bourgogne_franche_comte", culture_type="grandes_cultures", year_count=3,origin_year=2025):
        extracted_text_file_base_output_dir = os.path.join(self.base_directory_path, self.cfg["scraping"]["regions"][region]["output_dir_extracted_base_path"])
        scrapped_file_base_output_dir = Path(str(os.path.join(self.base_directory_path, self.cfg["scraping"]["regions"][region]["output_dir_pase_path"])))
        extracted_text = []
        start_year = origin_year - 1
        end_year = origin_year - year_count - 1
        print(scrapped_file_base_output_dir)
        for year in range(start_year, end_year, -1):
            bsv_dir = scrapped_file_base_output_dir / str(year)
            print(scrapped_file_base_output_dir)
            for fichier in bsv_dir.glob("*.pdf"):  # tous les PDF du dossier
                file_name = fichier.name.split('.')[0] + '.txt'
                with pdfplumber.open(fichier) as pdf:
                    for page in pdf.pages:
                        extracted_text.append(page.extract_text())
                        extracted_text.append(page.extract_table())
                year_dir = os.path.join(str(extracted_text_file_base_output_dir), str(year))
                os.makedirs(year_dir, exist_ok=True)
                with open(f'{year_dir}/{file_name}', "w") as text_file:
                    for chunk in extracted_text:
                        if chunk:
                            text_file.write(str(chunk))
                extracted_text.clear()


def main():
    setup_logging()
    logger.info("Démarrage du script extraction de texte")
    try:
        textExtractor = TextExtractor()
        textExtractor.extract_text_pdfplumber(
            region="bourgogne_franche_comte",
            culture_type="grandes_cultures",
            year_count=3,
            origin_year=2025
        )
        logger.info("Script terminé avec succès")
        return 0
    except KeyboardInterrupt:
        logger.warning("Interruption par l'utilisateur (Ctrl+C)")
        return 1
    except Exception as e:
        logger.error("Erreur fatale lors de l'exécution du script")
        logger.exception(e)
        return 1


if __name__ == "__main__":
    sys.exit(main())


