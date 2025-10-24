import os
import sys
import time
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from pathlib import Path

# Ajouter le dossier parent au PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.config_loader import ConfigLoader
from utils.file_utils import download_pdf
from utils.logger import setup_logging, get_logger

# Initialiser le logger pour ce module
logger = get_logger(__name__)


def retrieve_website_page(url: str) -> BeautifulSoup | None:
    try:
        logger.debug(f"Tentative de récupération de l'URL: {url}")
        response = requests.get(url, timeout=10)

        if response.status_code == 200:
            logger.info(f"Page récupérée avec succès: {url}")
            return BeautifulSoup(response.content, "lxml")
        else:
            logger.warning(f"Code HTTP inattendu {response.status_code} pour l'URL: {url}")
            print("Error retrieving website page")
            return None

    except requests.exceptions.Timeout:
        logger.error(f"Timeout lors de la récupération de l'URL: {url}")
        print(f"Timeout: {url}")
        return None

    except requests.exceptions.ConnectionError as e:
        logger.error(f"Erreur de connexion pour l'URL: {url}")
        logger.exception(e)
        print(e)
        return None

    except Exception as e:
        logger.error(f"Erreur inattendue lors de la récupération de l'URL: {url}")
        logger.exception(e)
        print(e)
        return None


class Scraping:


    def __init__(self):

        self.cfg = ConfigLoader().config
        self.base_directory_path = ConfigLoader().base_dir
        logger.info("Initialisation du scraper BSV")
        logger.debug(f"Répertoire de base: {self.base_directory_path}")

    def scrape_bsv(self, region="bourgogne_franche_comte", culture_type="grandes_cultures", year_count=3,
                   origin_year=2025):


        logger.info("=" * 80)
        logger.info("DÉBUT DU SCRAPING DRAAF")
        logger.info(f"Région: {region}")
        logger.info(f"Type de culture: {culture_type}")
        logger.info(f"Nombre d'années: {year_count}")
        logger.info(f"Année d'origine: {origin_year}")
        logger.info(f"Période couverte: {origin_year - year_count} à {origin_year - 1}")
        logger.info("=" * 80)

        # Construction des URLs
        draaf_url_website = self.cfg["scraping"]["draaf_url_website"]
        website_base_url = draaf_url_website + self.cfg["scraping"]["regions"][region]["previous_campaigns"][
            culture_type]
        base_output_dir = os.path.join(self.base_directory_path,
                                       self.cfg["scraping"]["regions"][region]["output_dir_pase_path"])

        logger.info(f"URL de base DRAAF: {draaf_url_website}")
        logger.info(f"URL complète: {website_base_url}")
        logger.info(f"Répertoire de sortie: {base_output_dir}")

        # Récupération de la page principale
        logger.info("Récupération de la page principale DRAAF...")
        draaf_html_content = retrieve_website_page(website_base_url)

        start_year = origin_year - 1
        end_year = origin_year - year_count - 1
        annual_bsv_pdf_links = {}
        pdf_docs = []

        if not draaf_html_content:
            logger.error("Impossible de récupérer la page principale")
            logger.error(f"URL tentée: {website_base_url}")
            print("Impossible de récupérer la page principale.")
            return

        logger.info("Page principale récupérée avec succès")

        # Recherche des liens par année
        logger.info(f"Recherche des liens pour les années {start_year} à {end_year}...")
        for year in range(start_year, end_year, -1):
            logger.info(f"Recherche des liens pour l'année {year}")
            links_found = 0

            for a_tag in draaf_html_content.find_all("a", href=True):
                if str(year) in a_tag["href"]:
                    annual_bsv_pdf_links[year] = urljoin(website_base_url, a_tag["href"])
                    links_found += 1

            if links_found > 0:
                logger.info(f"Trouvé {links_found} lien(s) pour l'année {year}")
            else:
                logger.warning(f"Aucun lien trouvé pour l'année {year}")

        logger.info(f"Total de {len(annual_bsv_pdf_links)} année(s) trouvée(s)")

        # Traitement de chaque année
        i = 1
        for year, annual_bsv_link in annual_bsv_pdf_links.items():
            logger.info("=" * 80)
            logger.info(f"TRAITEMENT DE L'ANNÉE {year} ({i}/{len(annual_bsv_pdf_links)})")
            logger.info("=" * 80)
            logger.info(f"URL de l'année: {annual_bsv_link}")
            print(f"{year}: {annual_bsv_link}")

            # Récupération de la page de l'année
            logger.info(f"Récupération de la page pour l'année {year}...")
            annual_bsv_page_html_content = retrieve_website_page(annual_bsv_link)

            if not annual_bsv_page_html_content:
                logger.warning(f"Impossible de récupérer la page pour l'année {year}, passage à l'année suivante")
                continue

            logger.info(f"Page de l'année {year} récupérée avec succès")

            # Collecte des liens PDF
            logger.info(f"Collecte des liens PDF pour l'année {year}...")
            pdf_count = 0

            for a_tag in annual_bsv_page_html_content.find_all("a", href=True):
                if a_tag["href"].endswith(".pdf"):
                    pdf_url = urljoin(website_base_url, a_tag["href"])
                    pdf_docs.append(pdf_url)
                    pdf_count += 1
                    logger.debug(f"PDF trouvé: {pdf_url}")

            logger.info(f"Nombre de PDF trouvés pour l'année {year}: {pdf_count}")

            # Création du répertoire de l'année
            year_dir = os.path.join(str(base_output_dir), str(year))
            os.makedirs(year_dir, exist_ok=True)
            logger.info(f"Répertoire créé/vérifié: {year_dir}")

            # Téléchargement des PDF
            logger.info(f"Début du téléchargement de {len(pdf_docs)} PDF pour l'année {year}...")
            downloaded = 0

            for idx, pdf_link in enumerate(pdf_docs, 1):
                file_name = pdf_link.split("/")[-1]
                logger.info(f"Téléchargement [{idx}/{len(pdf_docs)}]: {file_name}")

                success = download_pdf(pdf_link, str(year_dir), file_name)
                if success:
                    downloaded += 1
                    logger.debug(f"Téléchargement réussi: {file_name}")
                else:
                    logger.warning(f"Échec du téléchargement: {file_name}")

                time.sleep(1)

            logger.info(f"Téléchargement terminé pour l'année {year}: {downloaded}/{len(pdf_docs)} PDF téléchargés")
            print(f"bsv de {year}: {len(pdf_docs)}")

            pdf_docs.clear()
            i += 1

        logger.info("=" * 80)
        logger.info("SCRAPING TERMINÉ AVEC SUCCÈS")
        logger.info("=" * 80)


def main():

    setup_logging()
    logger.info("Démarrage du script de scraping BSV")
    logger.info("Plant Health NLP Analysis - Polytech Dijon")

    try:
        scraper = Scraping()
        scraper.scrape_bsv(
            region="bourgogne_franche_comte",
            culture_type="grandes_cultures",
            year_count=3
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