# src/flask/forms.py
from flask_wtf import FlaskForm
from wtforms import DecimalField, SubmitField, FileField
from wtforms.validators import DataRequired, NumberRange

class DataSplitForm(FlaskForm):
    # Campo per inserire la percentuale (tra 0 e 1) da destinare al training set
    train_ratio = DecimalField('Training Set Ratio (0-1)', 
                               default=0.7,
                               validators=[DataRequired(), NumberRange(min=0.0, max=1.0)])
    dataset_file = FileField('Carica un dataset CSV (opzionale)')
    submit = SubmitField('Applica modifiche')