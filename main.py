"""
ML-IDPS V4 - Main Entry Point
Stateful Behavioral Intrusion Detection and Prevention System

Features:
- Real-time packet capture (TShark)
- 26-feature extraction
- Three-layer detection: Rules + ML Ensemble (RF+IF+VAE) + Behavioral Memory
- Cross-flow IP reputation tracking
- Graduated firewall response
- Live statistics

Version: 4.0.0
"""
import asyncio
import logging
import sys
import warnings
import ipaddress
from typing import Optional, Any

# Suppress sklearn parallel.delayed warnings in this process AND joblib child processes
import os
os.environ['PYTHONWARNINGS'] = 'ignore::UserWarning'
warnings.filterwarnings('ignore', message='.*sklearn.utils.parallel.delayed.*')
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn.utils.parallel')
# Suppress ResourceWarnings for subprocess pipes
warnings.filterwarnings('ignore', category=ResourceWarning)
import json
import threading
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import yaml
import psutil
import uvicorn

# ── Performance constants ─────────────────────────────────────────────────────
# Cached timezone object — avoids repeated DB lookups on every threat event
_TZ_BAGHDAD = ZoneInfo('Asia/Baghdad')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))
from core.feature_extractor import FeatureExtractor
from core.threat_detector import ThreatDetector
from core.packet_capture import PacketCapture

from core.attack_injection import InjectionProducer
from core.threat_intel import ThreatIntelFeed
from api.database import Database
from api import injection_api
from core.threat_hunters.apt_detector import get_apt_detector
from core.threat_hunters.insider_threat import get_insider_detector

# Firewall prevention
try:
    from prevention.firewall import FirewallController
    FIREWALL_AVAILABLE = True
except ImportError:
    FIREWALL_AVAILABLE = False

# Logging setup
from logging.handlers import RotatingFileHandler

log_file = str(Path(__file__).parent / 'v2_ml_idps.log')
file_handler = RotatingFileHandler(log_file, maxBytes=100*1024*1024, backupCount=5, encoding='utf-8')
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

# Read log level from config.yaml at startup (before class instantiation)
def _get_log_level() -> int:
    try:
        _cfg_path = Path(__file__).parent / 'config' / 'config.yaml'
        with open(_cfg_path, 'r') as _f:
            _cfg = yaml.safe_load(_f)
        level_str = _cfg.get('logging', {}).get('level', 'INFO').upper()
        return getattr(logging, level_str, logging.INFO)
    except Exception:
        return logging.INFO

_log_level = _get_log_level()
logging.basicConfig(level=_log_level, handlers=[file_handler, stream_handler])
logger = logging.getLogger(__name__)


class MLIDPS_V4:
    """
    ML-IDPS Version 4.0
    Stateful Behavioral Intrusion Detection and Prevention System
    """
    
    # Private/local network ranges (RFC 1918 + link-local + loopback)
    PRIVATE_NETWORKS = [
        ipaddress.ip_network('10.0.0.0/8'),
        ipaddress.ip_network('172.16.0.0/12'),
        ipaddress.ip_network('192.168.0.0/16'),
        ipaddress.ip_network('127.0.0.0/8'),
        ipaddress.ip_network('169.254.0.0/16'),
        ipaddress.ip_network('fe80::/10'),
        ipaddress.ip_network('::1/128'),
        ipaddress.ip_network('fc00::/7'),
    ]
    
    # Get script directory for relative paths
    SCRIPT_DIR = Path(__file__).parent.resolve()
    
    def __init__(self, config_path: Optional[str] = None):
        logger.info("=" * 70)
        logger.info("ML-IDPS V4 SYSTEM STARTING")
        logger.info("=" * 70)
        
        # ── Runtime state (initialized in _setup_state below) ─────────────────
        self.running: bool = False
        self.total_packets: int = 0
        self.total_flows: int = 0
        self.threats_detected: int = 0
        self.rule_alerts_fired: int = 0   # Rule Engine ALERT verdicts (not ML-confirmed yet)
        self.start_time: Optional[datetime] = None
        self._last_cleanup_packets: int = 0
        self.stats_file: Path = Path()
        self.last_stats_update: Optional[datetime] = None
        self.blocked_ips: set = set()
        # Background stats writer queue (fix: decouple Disk I/O from detection loop)
        self._stats_queue: Optional[asyncio.Queue] = None
        # TShark watchdog counters
        self._tshark_restart_count: int = 0
        self._TSHARK_MAX_RESTARTS: int = 5
        
        # ── Component references ─────────────────────────────────────────────
        self.config: dict = {}
        self.model_config: dict = {}
        self.detector: Any = None
        self.feature_extractor: Any = None
        self.packet_capture: Any = None
        self.db: Any = None
        self.firewall: Any = None
        self.threat_intel: Optional[ThreatIntelFeed] = None
        
        # ── Injection references ─────────────────────────────────────────────
        self.injection_enabled: bool = False
        self.injection_queue: Optional[asyncio.Queue] = None
        self.injection_producer: Any = None
        self.injection_thread: Optional[threading.Thread] = None
        self.injection_config: dict = {}
        
        # Setup helpers
        self.config, self.model_config = self._setup_paths(config_path)
        self._setup_state()
        self._setup_components(self.model_config)
        self._setup_injection(self.model_config)
        
        # ── ConfigManager: thread-safe hot-reload with observer pattern ───────
        self._setup_config_manager(config_path)
        
        # Internal network monitoring mode
        net_cfg = self.config.get('network', {})
        self.internal_monitoring = net_cfg.get('internal_network_monitoring', False)
        self.my_ip = net_cfg.get('my_ip', '')
        
        if self.internal_monitoring:
            logger.info(f"⚠️ INTERNAL NETWORK MONITORING MODE — my_ip={self.my_ip}")
            # Add my_ip to firewall whitelist automatically
            if self.my_ip:
                whitelist = self.config.get('prevention', {}).get('whitelist', [])
                if self.my_ip not in whitelist:
                    whitelist.append(self.my_ip)
        
        logger.info("✅ ML-IDPS V4 initialized")
        
    def _setup_paths(self, config_path: Optional[str]) -> tuple:
        """Initialize paths and configuration"""
        if config_path is None:
            resolved: Path = self.SCRIPT_DIR / 'config' / 'config.yaml'
        else:
            resolved = Path(config_path)
            if not resolved.is_absolute():
                resolved = self.SCRIPT_DIR / resolved
        
        config = self._load_config(str(resolved))
        
        model_config = config.get('model', {})
        for key in ['rf_model_path', 'vae_model_path', 'if_model_path', 'scaler_path',
                     'anomaly_scaler_path', 'label_encoder_path', 'metadata_path', 'feature_config_path']:
            if key in model_config:
                path = Path(model_config[key])
                if not path.is_absolute():
                    model_config[key] = str(self.SCRIPT_DIR / path)
        return config, model_config

    def _setup_state(self):
        """Initialize (or reset) all runtime state variables.
        
        This is the single authoritative place for state initialization.
        Called once from __init__ — do NOT duplicate these assignments elsewhere.
        """
        self.running = False
        self.total_packets = 0
        self.total_flows = 0
        self.threats_detected = 0
        self.rule_alerts_fired = 0
        self.start_time = None
        self._last_cleanup_packets = 0
        
        self.stats_file = self.SCRIPT_DIR / 'data' / 'stats' / 'live_stats.json'
        self.stats_file.parent.mkdir(parents=True, exist_ok=True)
        
        # FIX UX: Remove stale stats file from previous run so UI doesn't show old numbers
        # while ML models are loading.
        if self.stats_file.exists():
            try:
                self.stats_file.unlink()
            except Exception:
                pass
                
        self.last_stats_update = None
        
        self.blocked_ips = set()

        # Prime psutil CPU counter — first call always returns 0.0%
        psutil.cpu_percent()

    def _setup_components(self, model_config: dict):
        """Initialize models, capturers, database, and firewall"""
        self.detector = ThreatDetector(self.config)
        self.feature_extractor = FeatureExtractor(
            flow_timeout=self.config['network'].get('flow_timeout', 5),
            config_path=model_config.get('feature_config_path')
        )
        self.packet_capture = PacketCapture(
            interface=self.config['network'].get('interface'),
            packet_filter=self.config['network'].get('capture_filter', 'tcp or udp'),
            buffer_size=self.config['network'].get('packet_buffer_size', 10000),
            use_tshark=self.config['network'].get('use_tshark', True)
        )
        
        db_path = self.config.get('database', {}).get('path', 'data/database/logs.db')
        if not Path(db_path).is_absolute():
            db_path = str(self.SCRIPT_DIR / db_path)
        self.db = Database(db_path)
        
        self.firewall = None
        if FIREWALL_AVAILABLE and self.config.get('prevention', {}).get('enabled', True):
            try:
                self.firewall = FirewallController(self.config.get('prevention', {}))
                logger.info("✅ FirewallController initialized (Graduated Response)")
                # Share with API layer so /api/block and /api/unblock
                # can apply real system firewall rules (not just DB updates)
                from api.shared_state import register_firewall
                register_firewall(self.firewall, self.blocked_ips)
            except Exception as e:
                logger.warning(f"FirewallController init failed: {e}")

        # ── Threat Intelligence Feed ─────────────────────────────────────────
        try:
            self.threat_intel = ThreatIntelFeed(
                cache_dir=str(self.SCRIPT_DIR / 'data' / 'threat_intel'),
                update_hours=6
            )
            # Try initial update (non-blocking — fails silently if offline)
            import threading
            def _ti_update():
                try:
                    stats = self.threat_intel.update()
                    logger.info(f"🛡️ Threat Intel ready: {stats.get('total_ips', 0):,} known malicious IPs")
                except Exception as e:
                    logger.warning(f"Threat Intel update failed (will use cache): {e}")
            threading.Thread(target=_ti_update, daemon=True).start()
        except Exception as e:
            logger.warning(f"Threat Intel init failed: {e}")
            self.threat_intel = None



    def _setup_injection(self, model_config: dict):
        """Initialize threat injection system settings"""
        inj = self.config.get('injection', {})
        self.injection_enabled = inj.get('enabled', False)
        self.injection_queue = None  # Instantiated safely in async start()
        self.injection_producer = None
        self.injection_thread = None
        if self.injection_enabled:
            dataset_path = inj.get('dataset_path', 'dataset/CIC-IDS2017')
            if not Path(dataset_path).is_absolute():
                candidate = self.SCRIPT_DIR / dataset_path
                if not candidate.exists():
                    candidate = self.SCRIPT_DIR.parent / dataset_path
                if not candidate.exists():
                    candidate = self.SCRIPT_DIR.parent / 'dataset' / 'CIC-IDS2017'
                dataset_path = str(candidate)
            self.injection_config = {
                'dataset_path': dataset_path,
                'metadata_path': model_config.get('metadata_path'),
                'attack_only': inj.get('attack_only', True),
                'max_rows': inj.get('max_rows') if inj.get('max_rows') is not None else 500,
                'samples_per_second': inj.get('samples_per_second', 5),
                'sample_strategy': inj.get('sample_strategy', 'first_n'),
                'samples_per_class': inj.get('samples_per_class', 80),
                'random_seed': inj.get('random_seed'),
                'max_load_rows': inj.get('max_load_rows'),
            }
            if not Path(dataset_path).exists():
                logger.warning(f"Injection dataset path does not exist: {dataset_path} — injection may load 0 rows")
            logger.info(
                f"Injection enabled: dataset={dataset_path}, rate={self.injection_config['samples_per_second']}/s, "
                f"samples={self.injection_config['max_rows']}, strategy={self.injection_config['sample_strategy']}"
            )

    def _setup_config_manager(self, config_path: Optional[str] = None):
        """Initialize ConfigManager and register live-reload observer callbacks.
        
        Each observer fires when its section is updated via the Settings API,
        applying changes to the relevant component in < 100ms without restart.
        """
        from core.config_manager import ConfigManager
        
        cfg_file = config_path or str(self.SCRIPT_DIR / 'config' / 'config.yaml')
        self.config_manager = ConfigManager(cfg_file)
        
        # ── Observer: Model thresholds ────────────────────────────────────────
        def _on_model_change(section: str, changes: dict):
            if self.detector:
                self.detector.apply_config_update('model', changes)
                logger.info(f"⚙️ Hot-reload [model]: {changes}")
        
        self.config_manager.subscribe('model', _on_model_change)
        
        # ── Observer: Prevention settings ─────────────────────────────────────
        def _on_prevention_change(section: str, changes: dict):
            if self.firewall:
                if 'enabled' in changes:
                    self.firewall.enabled = changes['enabled']
                if 'auto_block' in changes:
                    self.config['prevention']['auto_block'] = changes['auto_block']
                if 'whitelist' in changes:
                    self.config['prevention']['whitelist'] = changes['whitelist']
                    # Sync to FirewallController's set
                    self.firewall.whitelist = set(changes['whitelist'])
                if 'block_duration_hours' in changes:
                    self.config['prevention']['block_duration_hours'] = changes['block_duration_hours']

                # ═══════════════════════════════════════════════════════════════
                # NESTED CONFIG SUPPORT: graduated_response (Hot-Reload V2)
                # File watcher sends: {'graduated_response': {'enabled': True, ...}}
                # API sends flat:     {'level_1_action': 'ALERT', ...}
                # Both paths update FirewallController's internal state.
                # ═══════════════════════════════════════════════════════════════
                grad_changes = {}
                for k, v in changes.items():
                    if k == 'graduated_response' and isinstance(v, dict):
                        # Nested dict from file watcher — unpack
                        grad_changes.update(v)
                    elif k.startswith('level_') or k == 'graduated_response_enabled':
                        # Flat key from API
                        grad_changes[k] = v

                if grad_changes:
                    if 'enabled' in grad_changes:
                        self.firewall.graduated_response_enabled = bool(grad_changes['enabled'])
                    if 'level_3_duration_hours' in grad_changes:
                        self.firewall.level_durations[3] = int(grad_changes['level_3_duration_hours'])
                    if 'level_4_duration_hours' in grad_changes:
                        self.firewall.level_durations[4] = int(grad_changes['level_4_duration_hours'])
                    # Update config dict for persistence
                    self.config.setdefault('prevention', {}).setdefault('graduated_response', {}).update(grad_changes)
                    logger.info(f"  → GraduatedResponse updated: {list(grad_changes.keys())}")

            logger.info(f"⚙️ Hot-reload [prevention]: {list(changes.keys())}")
        
        self.config_manager.subscribe('prevention', _on_prevention_change)
        
        # ── Observer: Behavioral analysis ─────────────────────────────────────
        def _on_behavioral_change(section: str, changes: dict):
            if self.detector:
                self.detector.apply_config_update('behavioral_analysis', changes)
                logger.info(f"⚙️ Hot-reload [behavioral]: {changes}")
        
        self.config_manager.subscribe('behavioral_analysis', _on_behavioral_change)
        
        # ── Observer: Logging level ───────────────────────────────────────────
        def _on_logging_change(section: str, changes: dict):
            if 'level' in changes:
                new_level = getattr(logging, changes['level'].upper(), logging.INFO)
                logging.getLogger().setLevel(new_level)
                logger.info(f"⚙️ Hot-reload [logging]: level → {changes['level']}")
        
        self.config_manager.subscribe('logging', _on_logging_change)
        
        # ── Observer: Network settings ────────────────────────────────────────
        def _on_network_change(section: str, changes: dict):
            if 'internal_network_monitoring' in changes:
                self.internal_monitoring = changes['internal_network_monitoring']
            if 'my_ip' in changes:
                self.my_ip = changes['my_ip']
            if 'packet_buffer_size' in changes and self.packet_capture:
                self.packet_capture.buffer_size = changes['packet_buffer_size']
            logger.info(f"⚙️ Hot-reload [network]: {changes}")
        
        self.config_manager.subscribe('network', _on_network_change)
        
        # ── Observer: Injection system (Dynamic Lifecycle) ────────────────────
        def _on_injection_change(section: str, changes: dict):
            """Hot-reload observer for the injection section.
            
            Dynamically starts or stops the InjectionProducer thread:
              - Enable → create queue, producer, and background thread
              - Disable → stop producer, join thread, nullify queue
            
            The main _processing_loop already checks:
              if self.injection_queue is not None:
            So enabling/disabling takes effect on the next loop iteration.
            """
            if 'enabled' not in changes:
                # Only samples_per_second or other params changed — not toggling
                if 'samples_per_second' in changes:
                    self.injection_config['samples_per_second'] = changes['samples_per_second']
                    injection_api.update_injection_stat('samples_per_second', changes['samples_per_second'])
                logger.info(f"⚙️ Hot-reload [injection]: params updated {list(changes.keys())}")
                return

            should_enable = bool(changes['enabled'])
            currently_running = (self.injection_producer is not None and 
                                 self.injection_thread is not None and 
                                 self.injection_thread.is_alive())
            
            if should_enable and not currently_running:
                # ── ENABLE: Spin up injection dynamically ─────────────────
                logger.info("🔄 Hot-reload [injection]: ENABLING injection dynamically…")
                try:
                    # Rebuild config in case other params changed
                    inj = self.config_manager.get_section('injection')
                    dataset_path = inj.get('dataset_path', 'dataset/CIC-IDS2017')
                    if not Path(dataset_path).is_absolute():
                        candidate = self.SCRIPT_DIR / dataset_path
                        if not candidate.exists():
                            candidate = self.SCRIPT_DIR.parent / dataset_path
                        dataset_path = str(candidate)
                    
                    self.injection_config = {
                        'dataset_path': dataset_path,
                        'metadata_path': self.model_config.get('metadata_path'),
                        'attack_only': inj.get('attack_only', True),
                        'max_rows': inj.get('max_rows', 500),
                        'samples_per_second': inj.get('samples_per_second', 5),
                        'sample_strategy': inj.get('sample_strategy', 'first_n'),
                        'samples_per_class': inj.get('samples_per_class', 80),
                        'random_seed': inj.get('random_seed'),
                        'max_load_rows': inj.get('max_load_rows'),
                    }

                    self.injection_queue = asyncio.Queue()
                    loop = asyncio.get_event_loop()
                    cfg = self.injection_config
                    self.injection_producer = InjectionProducer(
                        self.injection_queue,
                        dataset_path=cfg['dataset_path'],
                        metadata_path=cfg.get('metadata_path'),
                        attack_only=cfg.get('attack_only', True),
                        max_rows=cfg.get('max_rows'),
                        samples_per_second=cfg.get('samples_per_second', 5),
                        event_loop=loop,
                        sample_strategy=cfg.get('sample_strategy', 'first_n'),
                        samples_per_class=cfg.get('samples_per_class', 80),
                        random_seed=cfg.get('random_seed'),
                        max_load_rows=cfg.get('max_load_rows'),
                    )
                    self.injection_thread = threading.Thread(
                        target=self.injection_producer.run_sync,
                        daemon=True,
                        name='InjectionProducer-HotReload',
                    )
                    self.injection_thread.start()
                    self.injection_enabled = True
                    injection_api.update_injection_stat('running', True)
                    injection_api.update_injection_stat('enabled', True)
                    injection_api.update_injection_stat('samples_per_second', cfg['samples_per_second'])
                    logger.info("✅ Injection started dynamically via hot-reload")
                except Exception as e:
                    logger.error(f"❌ Dynamic injection start failed: {e}")

            elif not should_enable and currently_running:
                # ── DISABLE: Tear down injection cleanly ──────────────────
                logger.info("🔄 Hot-reload [injection]: DISABLING injection…")
                try:
                    if self.injection_producer:
                        self.injection_producer.stop(timeout=5.0)
                    if self.injection_thread and self.injection_thread.is_alive():
                        self.injection_thread.join(timeout=5.0)
                    self.injection_queue = None
                    self.injection_producer = None
                    self.injection_thread = None
                    self.injection_enabled = False
                    injection_api.update_injection_stat('running', False)
                    injection_api.update_injection_stat('enabled', False)
                    logger.info("✅ Injection stopped cleanly via hot-reload")
                except Exception as e:
                    logger.error(f"❌ Dynamic injection stop failed: {e}")
            else:
                logger.info(f"⚙️ Hot-reload [injection]: no state change needed (enabled={should_enable}, running={currently_running})")
        
        self.config_manager.subscribe('injection', _on_injection_change)
        
        # ── Register with shared_state so API thread uses the SAME instance ──
        from api.shared_state import register_config_manager
        register_config_manager(self.config_manager)
        
        # ── Start file watcher: detects manual config.yaml edits in VS Code ──
        self.config_manager.start_watcher(poll_interval=2.0)
        
        logger.info("✅ ConfigManager initialized with 6 observer callbacks + file watcher")
    
    @staticmethod
    def _is_private_ip(ip_str: str) -> bool:
        """Check if IP belongs to private/local/reserved networks."""
        try:
            addr = ipaddress.ip_address(ip_str)
            return any(addr in net for net in MLIDPS_V4.PRIVATE_NETWORKS)
        except (ValueError, TypeError):
            return False
    
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML"""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"✅ Configuration loaded from {config_path}")
            return config
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            # Return default config
            return {
                'network': {'interface': None, 'flow_timeout': 5, 'use_tshark': True},
                'model': {
                    'rf_model_path': 'models/rf_model.pkl',
                    'if_model_path': 'models/if_model.pkl',
                    'vae_model_path': 'models/vae_model.pt',
                    'scaler_path': 'models/scaler.pkl',
                    'anomaly_scaler_path': 'models/anomaly_scaler.pkl',
                    'metadata_path': 'models/metadata.json',
                    'feature_config_path': 'config/features.json',
                    'rf_confidence_threshold': 0.65
                },
                'prevention': {'enabled': True, 'auto_block': True}
            }
    
    async def start(self) -> bool:
        """Start the detection system"""
        try:
            self.start_time = datetime.now()
            
            # Initialize database
            logger.info("Initializing database...")
            await self.db.initialize()
            await self.db.log_event('SYSTEM', 'ML-IDPS V4 starting', 'System initialization')
            
            # Load ML models
            logger.info("Loading ML models...")
            if not self.detector.load_models():
                logger.error("❌ Failed to load ML models!")
                logger.error("Run: python train_rf_ids2017.py && python train_anomaly_realnet.py")
                return False

            # Share the live ThreatDetector with the API layer
            # (allows ThreatHunter and SIEM webhook to access state_tracker)
            try:
                from api.shared_state import register_threat_detector
                register_threat_detector(self.detector)
            except Exception as _e:
                logger.warning(f"Could not register ThreatDetector with shared_state: {_e}")
            
            # Start packet capture
            logger.info("Starting packet capture...")
            if not self.packet_capture.start():
                logger.error("❌ Failed to start capture!")
                logger.error("Ensure TShark is installed and run as Administrator")
                return False
            
            self.running = True
            logger.info("=" * 70)
            logger.info("✅ ML-IDPS V4 RUNNING")
            logger.info("=" * 70)

            # ── FIX 2: Background stats writer — Disk I/O never blocks detection ──
            self._stats_queue = asyncio.Queue(maxsize=10)

            # ── Restore active blocks from Database ──
            active_blocks = await self.db.get_blocked_ips(include_expired=False)
            restored_count = 0
            for row in active_blocks:
                ip = row['ip_address']
                reason = row.get('reason', 'Restored on startup')
                level = row.get('block_level', 4)
                self.blocked_ips.add(ip)
                # BUGFIX: Guard against self.firewall being None
                # (happens when prevention.enabled=false in config.yaml)
                if self.firewall is not None and self.firewall.enabled:
                    self.firewall.block_manager.add_block(ip, reason, duration_hours=24, level=level)
                    # Re-apply OS-level rule just to be sure (it's idempotent)
                    self.firewall._apply_system_block(ip, reason)
                restored_count += 1
            if restored_count > 0:
                logger.info(f"🔄 Restored {restored_count} active blocks from database")

            await self.db.log_event('SYSTEM', 'ML-IDPS V4 started successfully', 'Detection active')
            
            # Start injection producer (background thread) if enabled
            if self.injection_enabled:
                self.injection_queue = asyncio.Queue()
                try:
                    loop = asyncio.get_running_loop()
                    cfg = self.injection_config
                    self.injection_producer = InjectionProducer(
                        self.injection_queue,
                        dataset_path=cfg['dataset_path'],
                        metadata_path=cfg.get('metadata_path'),
                        attack_only=cfg.get('attack_only', True),
                        max_rows=cfg.get('max_rows'),
                        samples_per_second=cfg.get('samples_per_second', 5),
                        event_loop=loop,
                        sample_strategy=cfg.get('sample_strategy', 'first_n'),
                        samples_per_class=cfg.get('samples_per_class', 80),
                        random_seed=cfg.get('random_seed'),
                        max_load_rows=cfg.get('max_load_rows'),
                    )
                    self.injection_thread = threading.Thread(
                        target=self.injection_producer.run_sync,
                        daemon=True,
                        name='InjectionProducer',
                    )
                    if self.injection_thread is None:
                        raise RuntimeError("Failed to create injection thread")
                    self.injection_thread.start()
                    injection_api.update_injection_stat('running', True)
                    injection_api.update_injection_stat('enabled', True)
                    injection_api.update_injection_stat('samples_per_second', cfg['samples_per_second'])
                    logger.info("✅ Attack injection started (CIC-IDS2017 → dashboard)")
                except Exception as e:
                    logger.warning(f"Injection start failed: {e}")

            # Run detection loop + watchdog + stats writer concurrently
            # return_exceptions=True: a crash in any sub-task is returned as a
            # value rather than re-raised, preventing one task from killing the others.
            results = await asyncio.gather(
                self._processing_loop(),
                self._tshark_watchdog(),
                self._stats_writer(),
                return_exceptions=True,
            )
            for r in results:
                if isinstance(r, Exception):
                    logger.error(f"Concurrent task failed: {r}")
            return True
            
        except Exception as e:
            logger.error(f"Start error: {e}")
            return False
    
    async def _processing_loop(self):
        """Main packet processing loop (packets + optional injection queue). Batch processing for throughput."""
        logger.info("Processing packets...")
        injection_batch_size = 15
        packet_batch_size = 10  # Default — adaptive below
        stats_interval_seconds = 2   # was 3 — faster dashboard refresh
        cleanup_interval_packets = 500
        _adaptive_scale = 1  # Tracks load pressure

        try:
            while self.running:
                features_list = []
                injected_flags = []
                has_processed = False
                
                # Adaptive batch sizing: grow under load to prevent queue overflow
                _q_size = self.packet_capture.get_queue_size() if hasattr(self.packet_capture, 'get_queue_size') else 0
                if _q_size > 1000:
                    packet_batch_size = 50
                elif _q_size > 200:
                    packet_batch_size = 25
                else:
                    packet_batch_size = 10
                # 1) Injection: drain up to batch_size from queue
                if self.injection_queue is not None:
                    for _ in range(injection_batch_size):
                        try:
                            features = self.injection_queue.get_nowait()  # type: ignore[union-attr]
                            features_list.append(features)
                            injected_flags.append(True)
                        except asyncio.QueueEmpty:
                            break
                
                # 2) Packets: batch drain (ALWAYS runs — injection must not starve real traffic)
                for pkt_idx in range(packet_batch_size):
                    # FIX: 1ms for first attempt (fast response), 10ms for subsequent (batching)
                    timeout = 0.001 if pkt_idx == 0 else 0.01
                    packet = self.packet_capture.get_packet(timeout=timeout)
                    if packet:
                        has_processed = True
                        self.total_packets += 1
                        features = self.feature_extractor.process_packet(packet)
                        if features:
                            features_list.append(features)
                            injected_flags.append(False)
                    else:
                        break
                
                # 3) Periodic cleanup — extract expired flows before deletion
                if self.total_packets - self._last_cleanup_packets >= cleanup_interval_packets:
                    self._last_cleanup_packets = self.total_packets
                    expired_features = self.feature_extractor.cleanup_old_flows()
                    for feat in expired_features:
                        features_list.append(feat)
                        injected_flags.append(False)
                    if self.total_packets % 2000 < packet_batch_size:
                        logger.info(f"Packets: {self.total_packets:,}, Flows: {self.total_flows:,}, Threats: {self.threats_detected}")
                
                # 4) Batch detect all collected features (10-50x faster than one-by-one)
                if features_list:
                    self.total_flows += len(features_list)
                    
                    for fd in features_list:
                        src_ip = fd.get('src_ip')
                        if src_ip and src_ip not in self.blocked_ips:
                            try:
                                get_apt_detector().record_event(
                                    src_ip, 
                                    'network_flow', 
                                    meta={'bytes': fd.get('Flow Bytes/s', 0)}
                                )
                                get_insider_detector().record_activity(
                                    src_ip, 
                                    bytes_transferred=int(float(fd.get('Flow Bytes/s', 0) or 0)), 
                                    resources_accessed=1
                                )
                            except Exception as e:
                                logger.error(f"Hunter record error: {e}")

                    results = self.detector.detect_batch(features_list, blocked_ips=self.blocked_ips)
                    for result, is_injected in zip(results, injected_flags):
                        if result.get('threat'):
                            # FIX M3: Don't inflate threat counter for already-blocked IPs
                            src_ip = result.get('src_ip', 'unknown')
                            if src_ip in self.blocked_ips:
                                continue
                            # Only increment threat counter if it passes LAN-anomaly safeguards and gets recorded
                            if await self._handle_threat(result, is_injected=is_injected):
                                self.threats_detected += 1
                        elif result.get('method', '').startswith('RULE:') and result.get('action') == 'ALERT':
                            # Rule Engine fired an ALERT (not confirmed as threat by ML)
                            # Count separately so it shows in dashboard
                            self.rule_alerts_fired += 1
                
                # Update stats (dashboard) — enqueue for background writer (non-blocking)
                now = datetime.now()
                _last = self.last_stats_update
                if _last is None or (now - _last).total_seconds() >= stats_interval_seconds:
                    self.last_stats_update = now
                    
                    try:
                        apt_findings = get_apt_detector().analyze_all()
                        for f in apt_findings:
                            asyncio.create_task(self.db.log_event('APT_HUNTER', f['description'], json.dumps(f)))
                    except Exception as e:
                        logger.error(f"APT Hunter error: {e}")
                    
                    if self._stats_queue is not None:
                        try:
                            self._stats_queue.put_nowait(self._build_stats_dict(now))
                        except asyncio.QueueFull:
                            pass  # Skip this cycle — writer is behind, no blocking
                    
                    # ═══════════════════════════════════════════════════════════
                    # PHASE 4: FP Monitoring — log threat ratio every stats cycle
                    # If threats > 20% of flows, something is wrong.
                    # ═══════════════════════════════════════════════════════════
                    if self.total_flows > 0:
                        threat_ratio = self.threats_detected / self.total_flows
                        active_flows = self.feature_extractor.get_active_flow_count()

                        # ── Change-only logging: log ONLY on meaningful events ──
                        # Track threats + blocked only (flows/pkts change every cycle)
                        _snap = (self.threats_detected, len(self.blocked_ips))
                        _prev_snap = getattr(self, '_last_log_snap', None)

                        if _prev_snap != _snap:
                            _prev = _prev_snap or _snap
                            _d_threats = self.threats_detected - _prev[0]
                            _d_blocked = len(self.blocked_ips) - _prev[1]
                            self._last_log_snap = _snap

                            # Build event description
                            _events = []
                            if _d_threats > 0:
                                _events.append(f"+{_d_threats} threat{'s' if _d_threats>1 else ''}")
                            if _d_blocked > 0:
                                _events.append(f"+{_d_blocked} blocked")
                            elif _d_blocked < 0:
                                _events.append(f"{_d_blocked} unblocked")
                            _event_str = '  '.join(_events) if _events else "state change"

                            if threat_ratio > 0.20:
                                logger.warning(
                                    f"⚠️  [{_event_str}]  "
                                    f"threats={self.threats_detected}  blocked={len(self.blocked_ips)}  "
                                    f"flows={self.total_flows:,}  pkts={self.total_packets:,}"
                                )
                            else:
                                logger.info(
                                    f"📡  [{_event_str}]  "
                                    f"threats={self.threats_detected}  blocked={len(self.blocked_ips)}  "
                                    f"flows={self.total_flows:,}  pkts={self.total_packets:,}"
                                )
                    
                    # ═══════════════════════════════════════════════════════════
                    # Memory Pressure Protection — prevent OOM under sustained attack
                    # SECURITY FIX: Do NOT disable VAE — an attacker can flood
                    # memory intentionally to bypass anomaly detection (Security Downgrade).
                    # Instead: apply aggressive rate-limiting to shed load while
                    # keeping the VAE shield active.
                    # ═══════════════════════════════════════════════════════════
                    _mem_pct = psutil.virtual_memory().percent
                    if _mem_pct > 95:
                        if not getattr(self, '_mem_emergency_active', False):
                            self._mem_emergency_active = True
                            # SECURITY: Never disable VAE — rate-limit sources instead
                            # Clear non-essential in-memory buffers to free RAM
                            if hasattr(self.detector, '_drift_window'):
                                self.detector._drift_window.clear()
                            # Aggressively drop stale flows to free memory
                            self.feature_extractor.cleanup_old_flows(max_age=30)
                            # Reduce flow table to a safe size
                            if hasattr(self.feature_extractor, '_flows'):
                                _flow_table = self.feature_extractor._flows
                                if len(_flow_table) > 5000:
                                    # Keep only the most recent 2500 flows
                                    _keys = list(_flow_table.keys())
                                    for _k in _keys[: len(_keys) - 2500]:
                                        _flow_table.pop(_k, None)
                            logger.critical(
                                f"🔴 Memory EMERGENCY ({_mem_pct}%)! "
                                "Dropping stale flows & clearing buffers. "
                                "VAE remains ACTIVE to maintain security posture."
                            )
                    elif _mem_pct > 93:
                        # Clear non-essential buffers
                        if hasattr(self.detector, '_drift_window'):
                            self.detector._drift_window.clear()
                        if not getattr(self, '_mem_warned', False):
                            self._mem_warned = True
                            logger.warning(f"⚠️ Memory HIGH ({_mem_pct}%)! Cleared drift buffer")
                    elif _mem_pct < 88 and getattr(self, '_mem_emergency_active', False):
                        # Recovered
                        self._mem_emergency_active = False
                        self._mem_warned = False
                        logger.info(f"✅ Memory recovered ({_mem_pct}%)")

                    # ═══════════════════════════════════════════════════════════
                    # FIX M2: Periodic StateTracker cleanup to free stale IPs
                    # Runs every stats cycle (~3s). Removes IPs idle > 10 min.
                    # ═══════════════════════════════════════════════════════════
                    self.detector.state_tracker.cleanup_stale(600)
                    # V5: Also clean stale IP events from Correlation Engine
                    if hasattr(self.detector, 'correlation_engine'):
                        self.detector.correlation_engine.cleanup_stale()
                    
                    # ═══════════════════════════════════════════════════════════
                    # FIX M6: Periodic firewall rule cleanup
                    # Removes expired netsh/iptables rules automatically
                    # ═══════════════════════════════════════════════════════════
                    if self.firewall and hasattr(self.firewall, 'cleanup_expired'):
                        expired = self.firewall.cleanup_expired()
                        if expired:
                            for ip in expired:
                                self.blocked_ips.discard(ip)
                            logger.info(f"Firewall cleanup: removed {len(expired)} expired rules")

                    # ═══════════════════════════════════════════════════════════
                    # FIX M7: Sync in-memory blocked_ips with DB every stats cycle
                    # Problem: when an operator unblocks an IP via the REST API
                    # (/api/unblock/<ip>), the DB entry is deleted but
                    # self.blocked_ips (Set) is never updated — the IDPS continues
                    # skipping all flows from that IP as if still blocked.
                    # Solution: rebuild the set from DB every ~3 s (stats interval).
                    # Cost: one async DB read per cycle — negligible vs. benefit.
                    # ═══════════════════════════════════════════════════════════
                    try:
                        active_db_blocks = await self.db.get_blocked_ips(include_expired=False)
                        db_blocked_set = {row['ip_address'] for row in active_db_blocks}
                        # Remove IPs that were unblocked via API (in DB but expired/deleted)
                        stale = self.blocked_ips - db_blocked_set
                        if stale:
                            for ip in stale:
                                self.blocked_ips.discard(ip)
                            logger.info(f"Synced blocked_ips: removed {len(stale)} API-unblocked IPs {stale}")
                    except Exception as _sync_err:
                        logger.debug(f"blocked_ips sync failed (non-critical): {_sync_err}")
                
                if not features_list and not has_processed:
                    await asyncio.sleep(0.005)
                else:
                    await asyncio.sleep(0)  # Yield control to event loop
                    
        except asyncio.CancelledError:
            logger.info("Processing cancelled")
        except Exception as e:
            logger.error(f"Processing error: {e}")
    
    async def _handle_threat(self, result: dict, is_injected: bool = False) -> bool:
        """Handle detected threat with graduated response - V4.1.

        Changes from V4.0:
          - Passes source_port, destination_port to add_alert()
          - Passes top_features (XAI), rule_id, rule_name, is_own_device to DB
          - Pushes live alert to injection_api buffer for immediate WebSocket broadcast
          - Returns True if processed/recorded, False if skipped by safeguards
        """
        try:
            if is_injected:
                inj = injection_api.get_injection_stats()
                injection_api.update_injection_stat('total_detected', inj['total_detected'] + 1)
                injection_api.update_injection_stat('current_attack_type', result.get('type', 'Unknown'))

            threat_type    = result.get('type', 'Unknown')
            confidence     = result.get('confidence', 0)
            action         = result.get('action', 'LOGGED')
            src_ip         = result.get('src_ip', 'unknown')
            dst_ip         = result.get('dst_ip', 'unknown')
            method         = result.get('method', 'Unknown')
            severity       = result.get('severity', 'medium')
            vae_score      = result.get('vae_score')
            rf_probability = result.get('rf_probability')

            # Forensic fields: stored in DB, not used in decision logic
            top_features  = result.get('top_features')          # XAI list (BLOCK only)
            src_port      = result.get('src_port') or result.get('Src Port')
            dst_port_raw  = result.get('dst_port') or result.get('Dst Port')
            dst_port      = int(dst_port_raw) if dst_port_raw is not None else None
            rule_id       = result.get('rule_id')               # e.g. 'R001_PORTSCAN'
            rule_name     = result.get('rule_name')
            # is_own_device: this machine is the traffic source (possible infection)
            is_own_device = (src_ip == self.my_ip)

            # SAFEGUARD 1: Skip anomaly-only alerts for private/local IPs
            # NOTE: Use startswith to also catch 'Zero-Day Anomaly: Behavioral' variants.
            # FP-FIX: LSTM_SEQUENCE alerts are also anomaly-only — the model is untrained
            # and produces noisy signals that must not bypass conservative escalation.
            is_anomaly_only = (
                threat_type.startswith('Zero-Day Anomaly')
                or method == 'LSTM_SEQUENCE'
            )
            if is_anomaly_only and not is_injected:
                if is_own_device:
                    logger.debug(f"Skipped self-anomaly alert: {src_ip}")
                    return False
                if not self.internal_monitoring and self._is_private_ip(src_ip):
                    logger.debug(f"Skipped anomaly alert for local IP: {src_ip}")
                    return False

            # SAFEGUARD 2: Zero-Day/LSTM - conservative escalation (ALERT only unless 95%+5 alerts)
            if is_anomaly_only:
                allow_block = False
                if confidence >= 0.95 and not self._is_private_ip(src_ip):
                    snap = self.detector.state_tracker.get_snapshot(src_ip)
                    if snap and snap.get('past_alerts', 0) >= 5:
                        allow_block = True
                        logger.warning(
                            f"Anomaly ESCALATED to BLOCK: conf={confidence:.2%}, "
                            f"past_alerts={snap.get('past_alerts')}, ip={src_ip}"
                        )
                if not allow_block:
                    action = 'ALERT'

            # CRITICAL: Skip already-blocked IPs (saves ~90% overhead)
            if src_ip in self.blocked_ips:
                if self.threats_detected % 100 == 0:
                    logger.debug(f"Skipped threat from blocked IP: {src_ip}")
                return False

            logger.warning(f"THREAT: {threat_type} from {src_ip} ({method}, conf: {confidence:.2%})")
            if is_own_device:
                # FP-FIX: Before treating dst_ip as C2, verify it is not a trusted CDN.
                # The LSTM and behavioral models can misclassify HTTPS to Google/Cloudflare
                # as beaconing. A CDN destination is never a real C2 server.
                _dst_is_cdn = (
                    hasattr(self, 'detector')
                    and hasattr(self.detector, 'rule_engine')
                    and self.detector.rule_engine.is_cdn_or_private(dst_ip)
                )
                if _dst_is_cdn:
                    logger.info(
                        f"  OWN DEVICE outbound to CDN {dst_ip} — "
                        f"NOT flagged as C2 (trusted range). Downgrading to ALERT."
                    )
                    action = 'ALERT'
                else:
                    logger.warning(
                        f"  OWN DEVICE IS SOURCE - possible infection. "
                        f"Blocking C2: {dst_ip}"
                    )

            final_action = action

            # ARCHITECTURAL SAFEGUARD: Never block this machine.
            # Infected device (Botnet) -> block the C2 destination, not self.
            offender_ip = dst_ip if is_own_device else src_ip

            # Firewall graduated response
            if self.firewall and self.firewall.enabled:
                try:
                    if is_anomaly_only:
                        final_action = 'ALERT'
                        logger.info("Response: ALERT only (anomaly - no escalation)")
                    else:
                        # V5: Pass risk_score to firewall for risk-based graduated response
                        risk_score = result.get('risk_score', 0.0)
                        response = self.firewall.handle_threat(
                            offender_ip, threat_type, confidence,
                            risk_score=risk_score
                        )
                        final_action = response.get('action', action)
                        level = response.get('level', 1)
                        threat_count = response.get('threat_count', 1)
                        logger.info(f"Response: Level {level}, Action: {final_action}, Count: {threat_count}, Risk: {risk_score:.1f}")

                        if final_action in ['BLOCK', 'TEMP_BLOCK']:
                            self.blocked_ips.add(offender_ip)
                            duration = 24 if final_action == 'BLOCK' else 1
                            await self.db.add_blocked_ip(
                                offender_ip, f"{threat_type} (Level {level})", level, duration
                            )
                            logger.warning(f"BLOCKED: {offender_ip} (Level {level})")
                            if is_own_device:
                                logger.warning(f"   (Redirected block from my_ip to C2: {dst_ip})")

                except Exception as e:
                    logger.error(f"Firewall response failed: {e}")

            # Fallback: simple blocking (RF-confirmed attacks only)
            elif self.config.get('prevention', {}).get('auto_block') and action == 'BLOCK' and not is_anomaly_only:
                self.blocked_ips.add(offender_ip)
                await self.db.add_blocked_ip(offender_ip, threat_type)
                logger.warning(f"BLOCKED: {offender_ip}")
                if is_own_device:
                    logger.warning(f"   (Redirected block from my_ip to C2: {dst_ip})")
                final_action = 'BLOCK'

            # Persist full forensic record to database
            await self.db.add_alert(
                source_ip=src_ip,
                destination_ip=dst_ip,
                attack_type=threat_type,
                confidence=confidence,
                action_taken=final_action,
                detection_method=method,
                severity=severity,
                source_port=src_port,
                destination_port=dst_port,
                vae_score=vae_score,
                if_score=result.get('if_score'),
                anomaly_score=result.get('anomaly_score'),
                rf_probability=rf_probability,
                top_features=top_features,
                rule_id=rule_id,
                rule_name=rule_name,
                is_own_device=is_own_device,
                # V5: Risk Score Engine
                risk_score=result.get('risk_score', 0.0),
                risk_level=result.get('risk_level', 'BENIGN'),
                correlation_pattern=(
                    result.get('correlation', {}).get('pattern_name')
                    if result.get('correlation') else None
                ),
            )

            # Push to WebSocket live buffer (< 1 second latency to dashboard)
            # Uses module-level _TZ_BAGHDAD constant — avoids per-call ZoneInfo lookup
            _now = datetime.now(_TZ_BAGHDAD).strftime('%Y-%m-%d %H:%M:%S')
            injection_api.push_live_alert({
                'timestamp':        _now,
                'source_ip':        src_ip,
                'destination_ip':   dst_ip,
                'source_port':      src_port,
                'destination_port': dst_port,
                'attack_type':      threat_type,
                'confidence':       round(confidence, 4),
                'severity':         severity,
                'action_taken':     final_action,
                'detection_method': method,
                'rule_id':          rule_id,
                'rule_name':        rule_name,
                'is_own_device':    is_own_device,
                'anomaly_score':    result.get('anomaly_score'),
                'if_score':         result.get('if_score'),
                'vae_score':        vae_score,
                # V5: Risk Score Engine
                'risk_score':       result.get('risk_score', 0.0),
                'risk_level':       result.get('risk_level', 'BENIGN'),
                'correlation':      result.get('correlation'),
            })
            return True

        except Exception as e:
            logger.error(f"Handle threat error: {e}")
            return False

    def _build_stats_dict(self, now: datetime) -> dict:
        """Build stats snapshot dict — pure CPU work, no I/O. Called from detection loop."""
        if self.injection_producer is not None:
            injection_api.update_injection_stat('total_injected', self.injection_producer.get_produced())
        inj = injection_api.get_injection_stats()
        if inj['total_injected'] > 0:
            injection_api.update_injection_stat(
                'detection_rate',
                (inj['total_detected'] / inj['total_injected']) * 100.0
            )
        uptime = (now - self.start_time).total_seconds() if self.start_time else 0
        rate = (inj['total_detected'] / inj['total_injected'] * 100.0) if inj['total_injected'] else 0.0
        return {
            'system_status': 'Active' if self.running else 'Inactive',
            'version': '4.0.0',
            'uptime_seconds': uptime,
            'total_packets': self.total_packets,
            'total_flows': self.total_flows,
            'threats_detected': self.threats_detected,
            'rule_alerts_fired': self.rule_alerts_fired,
            'blocked_ips': len(self.blocked_ips),
            'active_flows': self.feature_extractor.get_active_flow_count(),
            'cpu_usage': min(100.0, round(psutil.cpu_percent(interval=None), 1)),
            'memory_usage': min(100.0, round(psutil.virtual_memory().percent, 1)),
            'timestamp': now.isoformat(),
            'model_info': self.detector.get_model_info(),
            'injection_stats': {
                'running': inj['running'],
                'total': inj['total_injected'],
                'detected': inj['total_detected'],
                'detection_rate': round(rate, 1),
                'samples_per_sec': inj['samples_per_second'],
                'current_attack': inj['current_attack_type'] or 'None',
            },
        }

    async def _stats_writer(self):
        """FIX 2+M05: Background stats writer — drains _stats_queue, writes JSON via
        asyncio.to_thread so Disk I/O never blocks the event loop.
        """
        _last_written: bytes = b''

        def _write_sync(path, data: bytes):
            with open(path, 'wb') as f:
                f.write(data)

        while self.running:
            try:
                try:
                    stats = await asyncio.wait_for(self._stats_queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue
                # Compact JSON bytes — ~40% smaller, avoids re-encoding for dedup check
                payload = json.dumps(stats, separators=(',', ':')).encode()
                if payload == _last_written:
                    continue  # Skip identical write — nothing changed
                # Run blocking file write in thread pool (non-blocking for event loop)
                await asyncio.to_thread(_write_sync, self.stats_file, payload)
                _last_written = payload
            except Exception as e:
                logger.debug(f"Stats writer error (non-critical): {e}")

    async def _tshark_watchdog(self):
        """FIX 1: Watchdog that auto-restarts TShark if it crashes.
        Checks every 10 s whether tshark is still alive. Restarts up to
        _TSHARK_MAX_RESTARTS times before giving up with a critical log.
        """
        await asyncio.sleep(15)  # Grace period — let capture settle first
        while self.running:
            await asyncio.sleep(10)
            if not self.running:
                break
            try:
                alive = self.packet_capture.is_running()
            except Exception:
                alive = False
            if not alive:
                if self._tshark_restart_count >= self._TSHARK_MAX_RESTARTS:
                    logger.critical(
                        f"❌ TShark died and restart limit ({self._TSHARK_MAX_RESTARTS}) reached. "
                        "Capture is stopped — restart main.py manually."
                    )
                    break
                self._tshark_restart_count += 1
                logger.warning(
                    f"⚠️ TShark process died — auto-restarting "
                    f"(attempt {self._tshark_restart_count}/{self._TSHARK_MAX_RESTARTS})…"
                )
                try:
                    self.packet_capture.stop()
                    await asyncio.sleep(2)
                    ok = self.packet_capture.start()
                    if ok:
                        logger.info("✅ TShark restarted successfully")
                    else:
                        logger.error("❌ TShark restart failed — will retry in 10 s")
                except Exception as e:
                    logger.error(f"TShark restart exception: {e}")
    
    async def stop(self):
        """Stop the system"""
        logger.info("Stopping ML-IDPS V4...")
        self.running = False
        
        if self.injection_enabled:
            injection_api.update_injection_stat('running', False)
        if self.injection_producer:
            self.injection_producer.stop()
        if self.injection_thread is not None and self.injection_thread.is_alive():
            self.injection_thread.join(timeout=5.0)
        
        # ── FIX: Close DB BEFORE stopping packet capture / event loop ──────────
        # aiosqlite runs a worker thread; if the event loop closes before the
        # DB connection, the thread raises RuntimeError('Event loop is closed').
        # Closing the connection here drains pending futures while the loop
        # is still running, silencing the spurious error on shutdown.
        if hasattr(self, 'db') and self.db:
            try:
                await self.db.close()
            except Exception as _db_close_err:
                logger.debug(f"DB close during shutdown (non-critical): {_db_close_err}")
        
        self.packet_capture.stop()
        
        uptime = (datetime.now() - self.start_time).total_seconds() if self.start_time else 0
        
        logger.info("=" * 70)
        logger.info("FINAL STATISTICS")
        logger.info("=" * 70)
        logger.info(f"Uptime: {uptime:.0f} seconds")
        logger.info(f"Packets: {self.total_packets:,}")
        logger.info(f"Flows: {self.total_flows:,}")
        logger.info(f"Threats: {self.threats_detected}")
        logger.info(f"Blocked: {len(self.blocked_ips)}")
        logger.info("=" * 70)
        logger.info("✅ ML-IDPS V4 STOPPED")
    
    def get_stats(self) -> dict:
        """Get current statistics"""
        uptime = (datetime.now() - self.start_time).total_seconds() if self.start_time is not None else 0  # type: ignore[operator]
        return {
            'running': self.running,
            'uptime': uptime,
            'packets': self.total_packets,
            'flows': self.total_flows,
            'threats': self.threats_detected,
            'blocked': len(self.blocked_ips)
        }


def run_api_server():
    """Run API server in background thread"""
    from api.server import app
    config = uvicorn.Config(app, host="0.0.0.0", port=8000, log_level="warning", log_config=None)
    server = uvicorn.Server(config)
    server.run()


async def main():
    """Main entry point"""
    # Start API server in background thread
    api_thread = threading.Thread(target=run_api_server, daemon=True)
    api_thread.start()
    logger.info("API Server started on http://localhost:8000")
    
    # Start IDS
    idps = MLIDPS_V4()
    
    try:
        await idps.start()
    except KeyboardInterrupt:
        logger.info("\nInterrupt received")
    finally:
        await idps.stop()


if __name__ == "__main__":
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║                                                               ║
    ║         ML-IDPS V4 - Enterprise Network Security             ║
    ║         Rules + RF + IF + VAE Ensemble Detection              ║
    ║         26 Real-Time Features + Behavioral Memory             ║
    ║                                                               ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)
    
    if sys.version_info < (3, 9):
        print("❌ Python 3.9+ required")
        sys.exit(1)
    
    try:
        if sys.platform == 'win32':
            asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Shutdown by user (Ctrl+C)")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)
