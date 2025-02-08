# src/flask/forms.py
from flask_wtf import FlaskForm
from wtforms import DecimalField, SubmitField, FileField
from wtforms.validators import DataRequired, NumberRange

class DataSplitForm(FlaskForm):
    # Campo per inserire la percentuale (tra 0 e 1) da destinare al test size
    train_ratio = DecimalField('Test size (0-1)', 
                               default=0.7,
                               validators=[DataRequired(), NumberRange(min=0.0, max=1.0)])
    submit = SubmitField('Applica modifiche')