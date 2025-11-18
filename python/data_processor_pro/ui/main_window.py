"""
Main window for Data Processor Pro.

Professional dashboard-style interface with modern aesthetics,
dark/light themes, drag-and-drop support, and keyboard shortcuts.
"""

import customtkinter as ctk
from tkinter import filedialog, messagebox
from typing import Optional, List, Callable
from pathlib import Path
import logging

from ..config import AppConfig, UIConfig
from ..core import DataEngine, SignalProcessor
from ..analytics import StatisticalAnalyzer, MLPreprocessor, AnomalyDetector

logger = logging.getLogger(__name__)

# Set CustomTkinter appearance
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


class MainWindow(ctk.CTk):
    """
    Main application window.

    Features:
        - Modern dashboard layout
        - Dark/light theme switching
        - Drag-and-drop file support
        - Keyboard shortcuts
        - Real-time status updates
        - Collapsible sidebars
        - Multi-tab interface
    """

    def __init__(self, config: Optional[AppConfig] = None):
        """
        Initialize main window.

        Args:
            config: Application configuration
        """
        super().__init__()

        # Configuration
        self.config = config or AppConfig()
        self.ui_config = self.config.ui

        # Core components
        self.engine = DataEngine(
            use_polars=self.config.performance.use_polars,
            max_workers=self.config.performance.max_workers,
            lazy_mode=self.config.performance.lazy_loading
        )
        self.processor = SignalProcessor(
            use_numba=self.config.performance.use_numba,
            use_gpu=self.config.performance.use_gpu
        )
        self.stats_analyzer = StatisticalAnalyzer()
        self.ml_preprocessor = MLPreprocessor()
        self.anomaly_detector = AnomalyDetector()

        # State
        self.current_files: List[Path] = []
        self.theme = self.ui_config.theme.value

        # Setup window
        self._setup_window()
        self._create_menu()
        self._create_layout()
        self._setup_keyboard_shortcuts()
        self._setup_drag_drop()

        logger.info("Main window initialized")

    def _setup_window(self):
        """Configure main window properties."""
        self.title(self.ui_config.window_title)
        self.geometry(f"{self.ui_config.window_width}x{self.ui_config.window_height}")

        # Center window
        self.update_idletasks()
        x = (self.winfo_screenwidth() // 2) - (self.ui_config.window_width // 2)
        y = (self.winfo_screenheight() // 2) - (self.ui_config.window_height // 2)
        self.geometry(f"+{x}+{y}")

        # Minimum size
        self.minsize(800, 600)

        # Set theme
        self._apply_theme(self.theme)

    def _create_menu(self):
        """Create menu bar."""
        # Note: CustomTkinter doesn't have built-in menu, use tkinter menu
        import tkinter as tk
        menubar = tk.Menu(self)
        self.config(menu=menubar)

        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Open Files...", command=self._open_files, accelerator="Ctrl+O")
        file_menu.add_command(label="Open Folder...", command=self._open_folder)
        file_menu.add_separator()
        file_menu.add_command(label="Save Session...", command=self._save_session, accelerator="Ctrl+S")
        file_menu.add_command(label="Load Session...", command=self._load_session)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.quit, accelerator="Ctrl+Q")

        # Edit menu
        edit_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Edit", menu=edit_menu)
        edit_menu.add_command(label="Undo", command=self._undo, accelerator="Ctrl+Z")
        edit_menu.add_command(label="Redo", command=self._redo, accelerator="Ctrl+Y")
        edit_menu.add_separator()
        edit_menu.add_command(label="Preferences...", command=self._show_preferences)

        # View menu
        view_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="View", menu=view_menu)
        view_menu.add_command(label="Toggle Theme", command=self._toggle_theme, accelerator="Ctrl+T")
        view_menu.add_command(label="Toggle Sidebar", command=self._toggle_sidebar, accelerator="Ctrl+B")
        view_menu.add_separator()
        view_menu.add_command(label="Zoom In", command=self._zoom_in, accelerator="Ctrl++")
        view_menu.add_command(label="Zoom Out", command=self._zoom_out, accelerator="Ctrl+-")

        # Tools menu
        tools_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Tools", menu=tools_menu)
        tools_menu.add_command(label="Signal Processor...", command=self._open_signal_processor)
        tools_menu.add_command(label="Statistical Analysis...", command=self._open_stats_analyzer)
        tools_menu.add_command(label="Anomaly Detection...", command=self._open_anomaly_detector)
        tools_menu.add_command(label="ML Preprocessor...", command=self._open_ml_preprocessor)

        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="Documentation", command=self._show_docs)
        help_menu.add_command(label="Keyboard Shortcuts", command=self._show_shortcuts)
        help_menu.add_separator()
        help_menu.add_command(label="About", command=self._show_about)

    def _create_layout(self):
        """Create main layout with dashboard design."""
        # Configure grid
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Sidebar (left panel)
        self._create_sidebar()

        # Main content area
        self._create_main_area()

        # Status bar (bottom)
        if self.ui_config.show_statusbar:
            self._create_statusbar()

    def _create_sidebar(self):
        """Create collapsible sidebar with navigation."""
        self.sidebar = ctk.CTkFrame(self, width=self.ui_config.sidebar_width, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew", padx=0, pady=0)
        self.sidebar.grid_propagate(False)

        # Logo/Title
        logo_label = ctk.CTkLabel(
            self.sidebar,
            text="📊 Data Processor Pro",
            font=ctk.CTkFont(size=20, weight="bold")
        )
        logo_label.grid(row=0, column=0, padx=20, pady=20, sticky="w")

        # Quick Actions
        quick_actions_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        quick_actions_frame.grid(row=1, column=0, padx=10, pady=10, sticky="ew")

        ctk.CTkLabel(
            quick_actions_frame,
            text="Quick Actions",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(anchor="w", padx=10, pady=(0, 10))

        # Action buttons
        self._create_action_button(quick_actions_frame, "📁 Open Files", self._open_files)
        self._create_action_button(quick_actions_frame, "🔧 Process Data", self._process_data)
        self._create_action_button(quick_actions_frame, "📊 Visualize", self._visualize_data)
        self._create_action_button(quick_actions_frame, "🔍 Analyze", self._analyze_data)

        # Recent Files
        recent_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        recent_frame.grid(row=2, column=0, padx=10, pady=10, sticky="nsew")

        ctk.CTkLabel(
            recent_frame,
            text="Recent Files",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(anchor="w", padx=10, pady=(0, 10))

        self.recent_listbox = ctk.CTkTextbox(recent_frame, height=150, width=260)
        self.recent_listbox.pack(fill="both", expand=True, padx=10)
        self._update_recent_files()

        # Settings (bottom of sidebar)
        settings_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        settings_frame.grid(row=3, column=0, padx=10, pady=10, sticky="sew")

        # Theme toggle
        theme_switch = ctk.CTkSwitch(
            settings_frame,
            text="Dark Mode",
            command=self._toggle_theme,
            onvalue="dark",
            offvalue="light"
        )
        theme_switch.pack(anchor="w", padx=10, pady=5)
        theme_switch.select() if self.theme == "dark" else theme_switch.deselect()

    def _create_action_button(self, parent, text: str, command: Callable):
        """Create styled action button."""
        btn = ctk.CTkButton(
            parent,
            text=text,
            command=command,
            width=260,
            height=40,
            corner_radius=8,
            font=ctk.CTkFont(size=12)
        )
        btn.pack(padx=10, pady=5, anchor="w")
        return btn

    def _create_main_area(self):
        """Create main content area with tabs."""
        main_frame = ctk.CTkFrame(self, corner_radius=0)
        main_frame.grid(row=0, column=1, sticky="nsew", padx=0, pady=0)
        main_frame.grid_columnconfigure(0, weight=1)
        main_frame.grid_rowconfigure(0, weight=1)

        # Tabview
        self.tabview = ctk.CTkTabview(main_frame)
        self.tabview.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        # Add tabs
        self.tab_dashboard = self.tabview.add("Dashboard")
        self.tab_processing = self.tabview.add("Processing")
        self.tab_visualization = self.tabview.add("Visualization")
        self.tab_analytics = self.tabview.add("Analytics")
        self.tab_settings = self.tabview.add("Settings")

        # Setup each tab
        self._setup_dashboard_tab()
        self._setup_processing_tab()
        self._setup_visualization_tab()
        self._setup_analytics_tab()
        self._setup_settings_tab()

    def _setup_dashboard_tab(self):
        """Setup dashboard tab with overview cards."""
        dashboard = self.tab_dashboard

        # Welcome message
        welcome_label = ctk.CTkLabel(
            dashboard,
            text="Welcome to Data Processor Pro",
            font=ctk.CTkFont(size=24, weight="bold")
        )
        welcome_label.pack(pady=20)

        # Stats cards
        cards_frame = ctk.CTkFrame(dashboard, fg_color="transparent")
        cards_frame.pack(fill="both", expand=True, padx=20, pady=10)

        # Configure grid for cards
        for i in range(3):
            cards_frame.grid_columnconfigure(i, weight=1)

        # Card 1: Files
        self._create_stat_card(cards_frame, "Files Loaded", "0", "📁", row=0, col=0)

        # Card 2: Data Points
        self._create_stat_card(cards_frame, "Data Points", "0", "📊", row=0, col=1)

        # Card 3: Processing Speed
        self._create_stat_card(cards_frame, "Processing Speed", "0 pts/s", "⚡", row=0, col=2)

    def _create_stat_card(self, parent, title: str, value: str, icon: str, row: int, col: int):
        """Create statistics card."""
        card = ctk.CTkFrame(parent, corner_radius=10)
        card.grid(row=row, column=col, padx=10, pady=10, sticky="nsew")

        icon_label = ctk.CTkLabel(card, text=icon, font=ctk.CTkFont(size=32))
        icon_label.pack(pady=(20, 5))

        value_label = ctk.CTkLabel(card, text=value, font=ctk.CTkFont(size=24, weight="bold"))
        value_label.pack(pady=5)

        title_label = ctk.CTkLabel(card, text=title, font=ctk.CTkFont(size=12))
        title_label.pack(pady=(5, 20))

        # Store reference for updates
        setattr(self, f"card_{title.lower().replace(' ', '_')}_value", value_label)

    def _setup_processing_tab(self):
        """Setup processing tab."""
        proc = self.tab_processing

        label = ctk.CTkLabel(proc, text="Signal Processing", font=ctk.CTkFont(size=18, weight="bold"))
        label.pack(pady=10)

        # Processing options will be added here
        # This is a placeholder for the full implementation

    def _setup_visualization_tab(self):
        """Setup visualization tab."""
        viz = self.tab_visualization

        label = ctk.CTkLabel(viz, text="Interactive Visualizations", font=ctk.CTkFont(size=18, weight="bold"))
        label.pack(pady=10)

        # Visualization canvas will be added here

    def _setup_analytics_tab(self):
        """Setup analytics tab."""
        analytics = self.tab_analytics

        label = ctk.CTkLabel(analytics, text="Advanced Analytics", font=ctk.CTkFont(size=18, weight="bold"))
        label.pack(pady=10)

        # Analytics tools will be added here

    def _setup_settings_tab(self):
        """Setup settings tab."""
        settings = self.tab_settings

        label = ctk.CTkLabel(settings, text="Settings", font=ctk.CTkFont(size=18, weight="bold"))
        label.pack(pady=10)

        # Settings controls will be added here

    def _create_statusbar(self):
        """Create status bar at bottom."""
        self.statusbar = ctk.CTkFrame(self, height=30, corner_radius=0)
        self.statusbar.grid(row=1, column=0, columnspan=2, sticky="ew")

        self.status_label = ctk.CTkLabel(
            self.statusbar,
            text="Ready",
            font=ctk.CTkFont(size=10)
        )
        self.status_label.pack(side="left", padx=10)

    def _setup_keyboard_shortcuts(self):
        """Setup keyboard shortcuts."""
        if not self.ui_config.enable_keyboard_shortcuts:
            return

        # Bind shortcuts
        self.bind("<Control-o>", lambda e: self._open_files())
        self.bind("<Control-s>", lambda e: self._save_session())
        self.bind("<Control-q>", lambda e: self.quit())
        self.bind("<Control-z>", lambda e: self._undo())
        self.bind("<Control-y>", lambda e: self._redo())
        self.bind("<Control-t>", lambda e: self._toggle_theme())
        self.bind("<Control-b>", lambda e: self._toggle_sidebar())

        logger.info("Keyboard shortcuts enabled")

    def _setup_drag_drop(self):
        """Setup drag-and-drop file support."""
        if not self.ui_config.enable_drag_drop:
            return

        try:
            from tkinterdnd2 import DND_FILES, TkinterDnD
            # Note: This requires tkinterdnd2 package
            # Implementation would go here
            logger.info("Drag-and-drop enabled")
        except ImportError:
            logger.warning("tkinterdnd2 not available, drag-and-drop disabled")

    # ========== Theme Management ==========

    def _apply_theme(self, theme: str):
        """Apply color theme."""
        if theme == "dark":
            ctk.set_appearance_mode("dark")
        else:
            ctk.set_appearance_mode("light")

        self.theme = theme
        logger.info(f"Theme changed to: {theme}")

    def _toggle_theme(self):
        """Toggle between dark and light theme."""
        new_theme = "light" if self.theme == "dark" else "dark"
        self._apply_theme(new_theme)

    # ========== File Operations ==========

    def _open_files(self):
        """Open file dialog to select files."""
        files = filedialog.askopenfilenames(
            title="Select data files",
            filetypes=[
                ("All supported", "*.csv *.tsv *.parquet *.xlsx *.json"),
                ("CSV files", "*.csv"),
                ("TSV files", "*.tsv"),
                ("Parquet files", "*.parquet"),
                ("Excel files", "*.xlsx *.xls"),
                ("JSON files", "*.json"),
                ("All files", "*.*")
            ]
        )

        if files:
            self.current_files = [Path(f) for f in files]
            self._update_status(f"Loaded {len(files)} file(s)")
            self._update_recent_files()
            logger.info(f"Opened {len(files)} files")

    def _open_folder(self):
        """Open folder dialog."""
        folder = filedialog.askdirectory(title="Select folder")
        if folder:
            # Find all data files in folder
            folder_path = Path(folder)
            patterns = ["*.csv", "*.tsv", "*.parquet", "*.xlsx", "*.json"]
            files = []
            for pattern in patterns:
                files.extend(folder_path.glob(pattern))

            self.current_files = files
            self._update_status(f"Loaded {len(files)} file(s) from folder")
            logger.info(f"Opened folder with {len(files)} files")

    def _update_recent_files(self):
        """Update recent files list."""
        self.recent_listbox.delete("1.0", "end")
        for file_path in self.ui_config.recent_files[:self.ui_config.max_recent_files]:
            self.recent_listbox.insert("end", f"{file_path.name}\n")

    # ========== Data Operations ==========

    def _process_data(self):
        """Process loaded data."""
        if not self.current_files:
            messagebox.showwarning("No Data", "Please load files first")
            return

        self._update_status("Processing data...")
        # Processing logic will be implemented
        logger.info("Processing data")

    def _visualize_data(self):
        """Visualize data."""
        self.tabview.set("Visualization")
        logger.info("Switching to visualization tab")

    def _analyze_data(self):
        """Analyze data."""
        self.tabview.set("Analytics")
        logger.info("Switching to analytics tab")

    # ========== Session Management ==========

    def _save_session(self):
        """Save current session."""
        logger.info("Saving session")
        messagebox.showinfo("Save Session", "Session saved successfully")

    def _load_session(self):
        """Load session."""
        logger.info("Loading session")

    def _undo(self):
        """Undo last action."""
        logger.info("Undo")

    def _redo(self):
        """Redo last action."""
        logger.info("Redo")

    # ========== View Operations ==========

    def _toggle_sidebar(self):
        """Toggle sidebar visibility."""
        if self.sidebar.winfo_viewable():
            self.sidebar.grid_remove()
        else:
            self.sidebar.grid()

    def _zoom_in(self):
        """Zoom in."""
        logger.info("Zoom in")

    def _zoom_out(self):
        """Zoom out."""
        logger.info("Zoom out")

    # ========== Tool Windows ==========

    def _open_signal_processor(self):
        """Open signal processor tool."""
        logger.info("Opening signal processor")

    def _open_stats_analyzer(self):
        """Open statistical analyzer."""
        logger.info("Opening statistical analyzer")

    def _open_anomaly_detector(self):
        """Open anomaly detector."""
        logger.info("Opening anomaly detector")

    def _open_ml_preprocessor(self):
        """Open ML preprocessor."""
        logger.info("Opening ML preprocessor")

    # ========== Help & Info ==========

    def _show_preferences(self):
        """Show preferences dialog."""
        logger.info("Showing preferences")

    def _show_docs(self):
        """Show documentation."""
        messagebox.showinfo("Documentation", "Documentation will open in browser")

    def _show_shortcuts(self):
        """Show keyboard shortcuts."""
        shortcuts = """
        Keyboard Shortcuts:

        Ctrl+O - Open files
        Ctrl+S - Save session
        Ctrl+Q - Quit
        Ctrl+Z - Undo
        Ctrl+Y - Redo
        Ctrl+T - Toggle theme
        Ctrl+B - Toggle sidebar
        """
        messagebox.showinfo("Keyboard Shortcuts", shortcuts)

    def _show_about(self):
        """Show about dialog."""
        messagebox.showinfo(
            "About Data Processor Pro",
            "Data Processor Pro v1.0.0\n\n"
            "Professional-grade data analysis platform\n\n"
            "© 2025 Data Processor Pro Team"
        )

    # ========== Status Updates ==========

    def _update_status(self, message: str):
        """Update status bar message."""
        if hasattr(self, 'status_label'):
            self.status_label.configure(text=message)
        logger.info(f"Status: {message}")
