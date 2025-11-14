import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path


class ColoredFormatter(logging.Formatter):
    """Formatter avec couleurs pour la console"""

    # Codes ANSI pour les couleurs
    COLORS = {
        'DEBUG': '\033[36m',  # Cyan
        'INFO': '\033[32m',  # Vert
        'WARNING': '\033[33m',  # Jaune
        'ERROR': '\033[31m',  # Rouge
        'CRITICAL': '\033[35m',  # Magenta
        'RESET': '\033[0m'  # Reset
    }

    def format(self, record):
        # Ajouter la couleur selon le niveau
        levelname_colored = f"{self.COLORS.get(record.levelname, '')}{record.levelname}{self.COLORS['RESET']}"
        record.levelname = levelname_colored
        return super().format(record)


def setup_logging(config: dict = None, force_setup: bool = False):
    """
    Configure le système de logging pour tout le projet

    Args:
        config: Configuration complète du projet (optionnel)
        force_setup: Force la reconfiguration même si déjà fait
    """
    root_logger = logging.getLogger()

    # Éviter la double configuration (sauf si force_setup=True)
    if root_logger.handlers and not force_setup:
        return root_logger

    # Nettoyer les handlers existants si force_setup
    if force_setup:
        root_logger.handlers.clear()

    # Charger la config si non fournie
    if config is None:
        try:
            from utils.config_loader import ConfigLoader
            config = ConfigLoader().config
        except Exception:
            # Config par défaut si échec
            config = {
                'logging': {
                    'level': 'INFO',
                    'log_dir': '../logs',
                    'file_name': 'project.log',
                    'max_bytes': 10485760,
                    'backup_count': 5,
                    'console': True
                }
            }

    # Extraire la config logging
    logging_cfg = config.get('logging', {})
    level_str = logging_cfg.get('level', 'INFO').upper()
    log_dir = logging_cfg.get('log_dir', 'logs')
    file_name = logging_cfg.get('file_name', 'project.log')
    max_bytes = logging_cfg.get('max_bytes', 10485760)  # 10 MB par défaut
    backup_count = logging_cfg.get('backup_count', 5)
    use_console = logging_cfg.get('console', True)

    # Créer le dossier de logs
    log_dir_path = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) ,log_dir)
    log_dir_path.mkdir(parents=True, exist_ok=True)
    log_path = log_dir_path / file_name

    # Niveau de logging
    log_level = getattr(logging, level_str, logging.INFO)
    root_logger.setLevel(log_level)

    # Format des messages
    file_format = logging.Formatter(
        fmt='%(asctime)s | %(levelname)-8s | %(name)-30s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    console_format = ColoredFormatter(
        fmt='%(asctime)s | %(levelname)-8s | %(name)-25s | %(message)s',
        datefmt='%H:%M:%S'
    )

    # 1. File Handler avec rotation
    try:
        file_handler = RotatingFileHandler(
            log_path,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(file_format)
        root_logger.addHandler(file_handler)
    except Exception as e:
        print(f"ATTENTION: Impossible de créer le fichier de log: {e}", file=sys.stderr)

    # 2. Console Handler (optionnel)
    if use_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(console_format)
        root_logger.addHandler(console_handler)

    # Message de démarrage
    root_logger.info("=" * 80)
    root_logger.info("Plant Health NLP Analysis - Système de logging initialisé")
    root_logger.info(f"Niveau de log: {level_str}")
    root_logger.info(f"Fichier de log: {log_path}")
    root_logger.info(f"Console activée: {use_console}")
    root_logger.info("=" * 80)

    return root_logger


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)