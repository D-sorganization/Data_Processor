"""
Interactive visualization using Plotly for Data Processor Pro.

Provides professional, interactive charts with zoom, pan, hover,
and export capabilities.
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from typing import List, Optional, Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)


class InteractiveVisualizer:
    """
    Professional interactive visualization tools.

    Features:
        - Interactive line, scatter, bar, area plots
        - 3D visualizations
        - Heatmaps and correlation matrices
        - Subplots and multi-panel layouts
        - Spectrograms and time-frequency analysis
        - Export to HTML, PNG, SVG, PDF
        - Customizable themes and colors
    """

    def __init__(self, template: str = "plotly_dark"):
        """
        Initialize visualizer.

        Args:
            template: Plotly template ('plotly', 'plotly_dark', 'plotly_white', etc.)
        """
        self.template = template
        self.current_fig: Optional[go.Figure] = None

        logger.info(f"Interactive visualizer initialized with template: {template}")

    # ========== Basic Plots ==========

    def line_plot(self, x: np.ndarray, y_data: Dict[str, np.ndarray],
                 title: str = "Line Plot",
                 x_label: str = "X",
                 y_label: str = "Y",
                 show_legend: bool = True,
                 show_grid: bool = True) -> go.Figure:
        """
        Create interactive line plot.

        Args:
            x: X-axis data
            y_data: Dictionary of {name: y_values}
            title: Plot title
            x_label: X-axis label
            y_label: Y-axis label
            show_legend: Show legend
            show_grid: Show grid

        Returns:
            Plotly figure
        """
        fig = go.Figure()

        for name, y in y_data.items():
            fig.add_trace(go.Scatter(
                x=x,
                y=y,
                mode='lines',
                name=name,
                hovertemplate=f'{name}<br>X: %{{x}}<br>Y: %{{y}}<extra></extra>'
            ))

        fig.update_layout(
            title=title,
            xaxis_title=x_label,
            yaxis_title=y_label,
            template=self.template,
            showlegend=show_legend,
            xaxis=dict(showgrid=show_grid),
            yaxis=dict(showgrid=show_grid),
            hovermode='x unified'
        )

        self.current_fig = fig
        logger.info(f"Created line plot with {len(y_data)} signals")
        return fig

    def scatter_plot(self, x: np.ndarray, y: np.ndarray,
                    title: str = "Scatter Plot",
                    x_label: str = "X",
                    y_label: str = "Y",
                    color: Optional[np.ndarray] = None,
                    size: Optional[np.ndarray] = None) -> go.Figure:
        """
        Create interactive scatter plot.

        Args:
            x: X-axis data
            y: Y-axis data
            title: Plot title
            x_label: X-axis label
            y_label: Y-axis label
            color: Color values for points
            size: Size values for points

        Returns:
            Plotly figure
        """
        fig = go.Figure()

        marker_dict = {}
        if color is not None:
            marker_dict['color'] = color
            marker_dict['colorscale'] = 'Viridis'
            marker_dict['showscale'] = True
        if size is not None:
            marker_dict['size'] = size

        fig.add_trace(go.Scatter(
            x=x,
            y=y,
            mode='markers',
            marker=marker_dict if marker_dict else None,
            hovertemplate='X: %{x}<br>Y: %{y}<extra></extra>'
        ))

        fig.update_layout(
            title=title,
            xaxis_title=x_label,
            yaxis_title=y_label,
            template=self.template
        )

        self.current_fig = fig
        logger.info("Created scatter plot")
        return fig

    def bar_plot(self, categories: List[str], values: np.ndarray,
                title: str = "Bar Plot") -> go.Figure:
        """Create interactive bar plot."""
        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=categories,
            y=values,
            hovertemplate='%{x}<br>Value: %{y}<extra></extra>'
        ))

        fig.update_layout(
            title=title,
            template=self.template,
            xaxis_title="Category",
            yaxis_title="Value"
        )

        self.current_fig = fig
        return fig

    def area_plot(self, x: np.ndarray, y_data: Dict[str, np.ndarray],
                 title: str = "Area Plot",
                 stacked: bool = False) -> go.Figure:
        """Create interactive area plot."""
        fig = go.Figure()

        stackgroup = 'one' if stacked else None

        for name, y in y_data.items():
            fig.add_trace(go.Scatter(
                x=x,
                y=y,
                mode='lines',
                name=name,
                fill='tonexty' if stacked else 'tozeroy',
                stackgroup=stackgroup,
                hovertemplate=f'{name}<br>X: %{{x}}<br>Y: %{{y}}<extra></extra>'
            ))

        fig.update_layout(
            title=title,
            template=self.template,
            hovermode='x unified'
        )

        self.current_fig = fig
        return fig

    # ========== Advanced Plots ==========

    def plot_3d(self, x: np.ndarray, y: np.ndarray, z: np.ndarray,
               title: str = "3D Plot",
               plot_type: str = "scatter") -> go.Figure:
        """
        Create 3D visualization.

        Args:
            x, y, z: Coordinate data
            title: Plot title
            plot_type: 'scatter', 'surface', or 'line'

        Returns:
            Plotly figure
        """
        fig = go.Figure()

        if plot_type == "scatter":
            fig.add_trace(go.Scatter3d(
                x=x, y=y, z=z,
                mode='markers',
                marker=dict(
                    size=3,
                    color=z,
                    colorscale='Viridis',
                    showscale=True
                )
            ))
        elif plot_type == "surface":
            fig.add_trace(go.Surface(
                x=x, y=y, z=z,
                colorscale='Viridis'
            ))
        elif plot_type == "line":
            fig.add_trace(go.Scatter3d(
                x=x, y=y, z=z,
                mode='lines',
                line=dict(
                    color=z,
                    colorscale='Viridis',
                    width=4
                )
            ))

        fig.update_layout(
            title=title,
            template=self.template,
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z'
            )
        )

        self.current_fig = fig
        logger.info(f"Created 3D {plot_type} plot")
        return fig

    def heatmap(self, data: np.ndarray,
               x_labels: Optional[List[str]] = None,
               y_labels: Optional[List[str]] = None,
               title: str = "Heatmap",
               colorscale: str = "Viridis") -> go.Figure:
        """
        Create heatmap.

        Args:
            data: 2D array
            x_labels: X-axis labels
            y_labels: Y-axis labels
            title: Plot title
            colorscale: Color scale

        Returns:
            Plotly figure
        """
        fig = go.Figure()

        fig.add_trace(go.Heatmap(
            z=data,
            x=x_labels,
            y=y_labels,
            colorscale=colorscale,
            hovertemplate='X: %{x}<br>Y: %{y}<br>Value: %{z}<extra></extra>'
        ))

        fig.update_layout(
            title=title,
            template=self.template
        )

        self.current_fig = fig
        logger.info("Created heatmap")
        return fig

    def correlation_matrix(self, correlation_matrix: np.ndarray,
                          labels: List[str],
                          title: str = "Correlation Matrix") -> go.Figure:
        """Create correlation matrix heatmap."""
        # Add text annotations
        annotations = []
        for i in range(len(labels)):
            for j in range(len(labels)):
                annotations.append(
                    dict(
                        x=labels[j],
                        y=labels[i],
                        text=f"{correlation_matrix[i, j]:.2f}",
                        showarrow=False,
                        font=dict(color='white' if abs(correlation_matrix[i, j]) > 0.5 else 'black')
                    )
                )

        fig = self.heatmap(
            correlation_matrix,
            x_labels=labels,
            y_labels=labels,
            title=title,
            colorscale="RdBu"
        )

        fig.update_layout(annotations=annotations)
        return fig

    def box_plot(self, data_dict: Dict[str, np.ndarray],
                title: str = "Box Plot") -> go.Figure:
        """Create box plot for multiple datasets."""
        fig = go.Figure()

        for name, data in data_dict.items():
            fig.add_trace(go.Box(
                y=data,
                name=name,
                boxmean='sd'  # Show mean and std
            ))

        fig.update_layout(
            title=title,
            template=self.template,
            yaxis_title="Value"
        )

        self.current_fig = fig
        return fig

    def violin_plot(self, data_dict: Dict[str, np.ndarray],
                   title: str = "Violin Plot") -> go.Figure:
        """Create violin plot for distribution visualization."""
        fig = go.Figure()

        for name, data in data_dict.items():
            fig.add_trace(go.Violin(
                y=data,
                name=name,
                box_visible=True,
                meanline_visible=True
            ))

        fig.update_layout(
            title=title,
            template=self.template,
            yaxis_title="Value"
        )

        self.current_fig = fig
        return fig

    # ========== Signal Processing Visualizations ==========

    def spectrogram(self, frequencies: np.ndarray,
                   times: np.ndarray,
                   spectrogram: np.ndarray,
                   title: str = "Spectrogram") -> go.Figure:
        """
        Create spectrogram visualization.

        Args:
            frequencies: Frequency values
            times: Time values
            spectrogram: 2D spectrogram data
            title: Plot title

        Returns:
            Plotly figure
        """
        fig = go.Figure()

        fig.add_trace(go.Heatmap(
            x=times,
            y=frequencies,
            z=10 * np.log10(spectrogram + 1e-10),  # Convert to dB
            colorscale='Jet',
            colorbar=dict(title='Power (dB)')
        ))

        fig.update_layout(
            title=title,
            xaxis_title='Time',
            yaxis_title='Frequency (Hz)',
            template=self.template
        )

        self.current_fig = fig
        logger.info("Created spectrogram")
        return fig

    def power_spectrum(self, frequencies: np.ndarray,
                      power: np.ndarray,
                      title: str = "Power Spectrum",
                      log_scale: bool = True) -> go.Figure:
        """Create power spectral density plot."""
        fig = go.Figure()

        y_data = 10 * np.log10(power) if log_scale else power

        fig.add_trace(go.Scatter(
            x=frequencies,
            y=y_data,
            mode='lines',
            fill='tozeroy'
        ))

        fig.update_layout(
            title=title,
            xaxis_title='Frequency (Hz)',
            yaxis_title='Power (dB)' if log_scale else 'Power',
            template=self.template
        )

        self.current_fig = fig
        return fig

    # ========== Subplots ==========

    def create_subplots(self, rows: int, cols: int,
                       subplot_titles: Optional[List[str]] = None,
                       shared_xaxes: bool = False,
                       shared_yaxes: bool = False) -> go.Figure:
        """
        Create subplot figure.

        Args:
            rows: Number of rows
            cols: Number of columns
            subplot_titles: List of subplot titles
            shared_xaxes: Share x-axes
            shared_yaxes: Share y-axes

        Returns:
            Plotly figure with subplots
        """
        fig = make_subplots(
            rows=rows,
            cols=cols,
            subplot_titles=subplot_titles,
            shared_xaxes=shared_xaxes,
            shared_yaxes=shared_yaxes
        )

        fig.update_layout(template=self.template)

        self.current_fig = fig
        logger.info(f"Created subplot grid: {rows}x{cols}")
        return fig

    # ========== Export Methods ==========

    def export_html(self, filename: str,
                   fig: Optional[go.Figure] = None,
                   include_plotlyjs: str = 'cdn') -> None:
        """
        Export plot to HTML.

        Args:
            filename: Output filename
            fig: Figure to export (uses current if None)
            include_plotlyjs: How to include plotly.js
        """
        if fig is None:
            fig = self.current_fig

        if fig is None:
            raise ValueError("No figure to export")

        fig.write_html(filename, include_plotlyjs=include_plotlyjs)
        logger.info(f"Exported to HTML: {filename}")

    def export_image(self, filename: str,
                    fig: Optional[go.Figure] = None,
                    width: int = 1920,
                    height: int = 1080,
                    format: str = 'png') -> None:
        """
        Export plot to static image.

        Args:
            filename: Output filename
            fig: Figure to export (uses current if None)
            width: Image width
            height: Image height
            format: Image format ('png', 'svg', 'pdf')
        """
        if fig is None:
            fig = self.current_fig

        if fig is None:
            raise ValueError("No figure to export")

        fig.write_image(filename, width=width, height=height, format=format)
        logger.info(f"Exported to {format.upper()}: {filename}")

    # ========== Utility Methods ==========

    def set_template(self, template: str) -> None:
        """Change plot template."""
        self.template = template
        if self.current_fig:
            self.current_fig.update_layout(template=template)
        logger.info(f"Template changed to: {template}")

    def show(self, fig: Optional[go.Figure] = None) -> None:
        """Display plot in browser."""
        if fig is None:
            fig = self.current_fig

        if fig is None:
            raise ValueError("No figure to show")

        fig.show()

    def get_current_figure(self) -> Optional[go.Figure]:
        """Get current figure."""
        return self.current_fig
