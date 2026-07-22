# Concrete-Filled-Steel-Tube-Columns-Multi-Objective-Optimization
This tool is developed as part of the research project "Explainable deep learning-driven multi-objective design optimization of stub and slender CFST columns under axial compression.". The study provides a robust and efficient way to predict the ultimate strength of CFST columns subjected to axial load and evaluate the multi-objective optimization analysis against the cost function.

1. Download all files from this repository to your local computer, including the python file app.py and resource folder.
2. Install the required libraries in Python by running this command in your terminal: pip install streamlit xgboost pandas numpy matplotlib joblib shap
3. Open your Command Prompt (CMD) or Terminal and navigate to the project directory where you saved the files.
4. Execute the command: streamlit run app.py.
5. Once the command is executed, your web browser will automatically open the interface.
6. Choose the cross-section type of CFST column to predict, define input parameters.
7. The appropriate model of slenderness will be selected based on slenderness ratio.
8. The results include predicted ultimate strength and the demonstration of the optimization performance within generated Pareto fronts.
