# GeoStackingPredictor
GeoStackingPredictor 
Description: GeoStackingPredictor is an advanced, interpretable 3D geological modeling tool developed using the Stacking Ensemble Learning approach. It is specifically designed for predicting lithology in unknown subterranean regions based on limited drilling data. By harnessing the stacking technique in ensemble learning, it merges multiple classifiers to ensure superior prediction accuracy, especially for stratums with smaller sample sizes. Furthermore, the tool comes equipped with a 3D geological visualization function, enabling users to deeply analyze and explore the predicted geological structures in a spatial context. For enhanced interpretability, a mechanism analysis based on SHAP is integrated, allowing users to not only visualize predictions but also comprehend the reasoning behind them.
Installation Guide:
git clone https://github.com/Lukacut/GeoStackingPredictor.git
cd GeoStackingPredictor
pip install -r requirements.txt  # Add this file to your repo with the required packages
Usage:
1.Data Training: Begin by loading your drilling data. Ensure it's in CSV format, featuring spatial coordinates columns 'x', 'y', 'z', and a 'lithology' column for rock-type classification. The model will be trained using this dataset.
2.Prediction Input: Once the model is trained, input the spatial coordinates where you want to predict the lithology.
3.Visualization & Analysis: After obtaining predictions, utilize the 3D geological visualization to view the forecasted geological structures. Dive deep into the predicted areas, exploring the geological intricacies.
4.Mechanism Analysis: Leverage the integrated SHAP mechanism analysis to gain insights into why certain predictions were made, understanding the underlying drivers and influences.
Key Features:
Stacking Ensemble Learning: Combines multiple machine learning models, ensuring optimal prediction accuracy.
3D Geological Visualization: Offers an immersive representation of predicted lithology in 3D space, providing a richer geological understanding.
SHAP Mechanism Analysis: Delivers in-depth explanations using SHAP, revealing not only the prediction outcomes but also the rationale behind them.
Dependencies: numpy, pandas, matplotlib, scikit-learn, xgboost, shap
Contributors: Lukacdut
