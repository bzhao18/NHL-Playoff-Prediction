# This function is for visualizing data and plotting it in every possible way in the second dimension.
# Note that it will take some time to run on data that has many rows and columns.
# Note that plotting large amounts of data will also require zooming in to be able to read the data.

import numpy as np
import matplotlib.pyplot as plt


def visualizeComponents(data: np.array, labels=None):
    # Get number of component (features) from data, this will determine the grid for our figure
    row, components = np.shape(data)

    # Will have some extra empty plots but they will be removed later
    fig, axes = plt.subplots(components - 1, components - 1)
    figSize = 4.5 * components - 1
    # Important so that scaling adapts to the number of plots we are creating
    fig.set_size_inches(figSize, figSize)
    fig.suptitle('Data Visualization', fontsize=30, fontweight='bold')
    fig.set_dpi(100)

    # Iterate over principal components and plot them
    for i in range(components - 1):
        for j in range(i + 1, components):
            # Get two features to be plotted from data
            xPC = data[:, i]
            yPC = data[:, j]

            # Create labels for the x and y axis.
            # When the list is none, we can assume that we are plotting data for PCA,
            # which can be labeled in terms of principal components
            if labels is None:
                xLabel = 'PC' + str(i)
                yLabel = 'PC' + str(j)
            else:
                xLabel = labels[i]
                yLabel = labels[j]

            # Gives random colors to plots
            color = np.random.rand(1, 4)
            # Setting for the alpha (transparency)
            # Having this at a lower rate will allow us more easily recognize overlapping data points and density patterns
            color[0, 3] = 0.1
            # color = 'b' uncomment this to make all the same color blue

            # Create scatter plot and add labels to it
            axes[i, j - 1 - i].scatter(xPC, yPC, s=5, c=color)
            axes[i, j - 1 - i].set_xlabel(xLabel, fontsize=12.5, fontweight='bold')
            axes[i, j - 1 - i].set_ylabel(yLabel, fontsize=12.5, fontweight='bold')

    # This is for deleting empty scatter plots
    for i in range(components - 1):
        for j in range(components - 1):
            if not axes[i, j].collections:
                plt.delaxes(axes[i, j])

    plt.show()
