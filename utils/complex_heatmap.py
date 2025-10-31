import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

class ComplexHeatmap:
    """
    Visualize complex number matrices with heatmaps for real, imaginary, absolute values, and optional phase.
    
    # Collaborated with Grok 3
    """
    
    def __init__(self, cmap_diverging='RdBu', cmap_absolute='Blues', cmap_phase='hsv', figsize_three=(15, 5), figsize_four=(20, 5)):
        """
        Initialize visualization settings.
        
        Args:
            cmap_diverging (str): Colormap for real and imaginary parts (default: 'RdBu').
            cmap_absolute (str): Colormap for absolute value (default: 'Blues').
            cmap_phase (str): Colormap for phase (default: 'hsv').
            figsize_three (tuple): Figure size for three subplots (default: (15, 5)).
            figsize_four (tuple): Figure size for four subplots (default: (20, 5)).
        """
        self.cmap_diverging = cmap_diverging
        self.cmap_absolute = cmap_absolute
        self.cmap_phase = cmap_phase
        self.figsize_three = figsize_three
        self.figsize_four = figsize_four
    
    def plot(self, complex_matrix, annot=True, title=None, save_path=None, show_phase=True):
        """
        Plot heatmaps for a complex matrix: real part, imaginary part, absolute value, and optional phase.
        
        Args:
            complex_matrix (np.ndarray): 2D array of complex numbers.
            annot (bool): Annotate cells with values (default: True).
            title (str, optional): Title for the entire plot (default: None).
            save_path (str, optional): File path to save the plot (default: None).
            show_phase (bool): Include phase heatmap (default: True).
        """
        real_part = np.real(complex_matrix)
        imag_part = np.imag(complex_matrix)
        abs_value = np.abs(complex_matrix)
        
        vmin = min(real_part.min(), imag_part.min())
        vmax = max(real_part.max(), imag_part.max())
        
        if show_phase:
            phase = np.angle(complex_matrix, deg=True)
            fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=self.figsize_four)
        else:
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=self.figsize_three)
        
        sns.heatmap(real_part, annot=annot, cmap=self.cmap_diverging, vmin=vmin, vmax=vmax, center=0, ax=ax1)
        ax1.set_title('Real Part')
        
        sns.heatmap(imag_part, annot=annot, cmap=self.cmap_diverging, vmin=vmin, vmax=vmax, center=0, ax=ax2)
        ax2.set_title('Imaginary Part')
        
        sns.heatmap(abs_value, annot=annot, cmap=self.cmap_absolute, vmin=0, vmax=abs_value.max(), ax=ax3)
        ax3.set_title('Absolute Value')
        
        if show_phase:
            sns.heatmap(phase, annot=annot, cmap=self.cmap_phase, vmin=-180, vmax=180, ax=ax4)
            ax4.set_title('Phase (degrees)')
        
        if title:
            fig.suptitle(title, fontsize=16, y=0.975)
        
        plt.tight_layout()
        fig.subplots_adjust(top=0.85)
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        
        plt.show()

if __name__ == "__main__":
    complex_matrix = np.array([[1+2j, 3-1j, 0+1j], [-1+0j, 2+2j, 1-3j], [0-1j, -2+1j, 3+0j]])
    heatmap = ComplexHeatmap()
    heatmap.plot(complex_matrix, title="Complex Matrix 1 with Phase", save_path="matrix1_phase.png", show_phase=True)
    heatmap.plot(complex_matrix, title="Complex Matrix 1 without Phase", save_path="matrix1_no_phase.png", show_phase=False)
    another_matrix = np.array([[0+1j, 2-2j], [1+0j, -1+1j]])
    heatmap.plot(another_matrix, title="Complex Matrix 2 with Phase", save_path="matrix2_phase.png", show_phase=True)