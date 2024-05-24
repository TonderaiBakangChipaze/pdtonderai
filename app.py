from dash import dcc, html, Dash, callback, Input, Output, State, ctx
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd

# Seed for reproducibility
np.random.seed(42)

# Data
sports = ['Soccer', 'Basketball', 'Tennis', 'Baseball', 'Cricket', 'Rugby', 'Hockey', 'Swimming', 'Athletics', 'Gymnastics']
continents = ['Europe', 'North America', 'Asia', 'South America', 'Australia', 'Africa', 'Antarctica']
countries_by_continent = {
    'Europe': ['Germany', 'France', 'Italy', 'Spain', 'UK', 'Russia', 'Netherlands', 'Belgium', 'Sweden', 'Switzerland'],
    'North America': ['USA', 'Canada', 'Mexico', 'Cuba', 'Guatemala', 'Honduras', 'Panama', 'Jamaica', 'Costa Rica', 'Haiti'],
    'Asia': ['China', 'Japan', 'India', 'South Korea', 'Indonesia', 'Philippines', 'Vietnam', 'Thailand', 'Malaysia', 'Singapore'],
    'South America': ['Brazil', 'Argentina', 'Chile', 'Peru', 'Colombia', 'Venezuela', 'Uruguay', 'Paraguay', 'Bolivia', 'Ecuador'],
    'Australia': ['Australia', 'New Zealand', 'Papua New Guinea', 'Fiji', 'Samoa', 'Tonga', 'Solomon Islands', 'Vanuatu', 'Kiribati', 'Nauru'],
    'Africa': ['South Africa', 'Nigeria', 'Egypt', 'Kenya', 'Ghana', 'Ethiopia', 'Tanzania', 'Uganda', 'Algeria', 'Morocco'],
    'Antarctica': []
}

countries = [country for continent in countries_by_continent.values() for country in continent]

# Time intervals between 8:00 and 20:00, 2-hour intervals
time_intervals = ['08:00', '10:00', '12:00', '14:00', '16:00', '18:00', '20:00']
ages = ['Under 18', '18-35', '35-50', '50 and above']
genders = ['Male', 'Female', 'Other']

# Generate data
data = {
    'Event': [],
    'Continent': [],
    'Country': [],
    'Views': [],
    'Visits': [],
    'Time': [],
    'Age': [],
    'Gender': []
}

for continent, countries in countries_by_continent.items():
    for country in countries:
        for event in sports:
            for age in ages:
                for gender in genders:
                    data['Event'].append(event)
                    data['Continent'].append(continent)
                    data['Country'].append(country)
                    data['Views'].append(np.random.randint(500000, 900000))  # In thousands
                    data['Visits'].append(np.random.randint(1000000, 1500000))  # In thousands
                    data['Time'].append(np.random.choice(time_intervals))
                    data['Age'].append(age)
                    data['Gender'].append(gender)

# Convert data to DataFrame and reduce its size by half
df = pd.DataFrame(data).sample(frac=0.5, random_state=42)

# External stylesheet
external_stylesheets = ['https://stackpath.bootstrapcdn.com/bootswatch/4.5.2/flatly/bootstrap.min.css']

app = Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server

app.layout = html.Div([
    html.H3("2024 Paris Fun Olympic Games Dashboard", className="text-center my-4"),

    html.Div([
        html.Div([
            html.H4("Filter Options"),
            dcc.Dropdown(
                id='continent-dropdown',
                options=[{'label': c, 'value': c} for c in df['Continent'].unique()],
                value=['Asia'],  # Default continent
                placeholder="Select Continent",
                multi=True
            ),
            dcc.Dropdown(
                id='country-dropdown',
                options=[],
                placeholder="Select Country",
                value=None,
                multi=True
            ),
            dcc.Dropdown(
                id='sport-dropdown',
                options=[{'label': sport, 'value': sport} for sport in df['Event'].unique()],
                value=None,
                placeholder="Select Sport",
                multi=True
            ),
            dcc.Dropdown(
                id='age-dropdown',
                options=[{'label': age, 'value': age} for age in df['Age'].unique()],
                value=None,
                placeholder="Select Age",
                multi=True
            ),
            dcc.Dropdown(
                id='gender-dropdown',
                options=[{'label': gender, 'value': gender} for gender in df['Gender'].unique()],
                value=None,
                placeholder="Select Gender",
                multi=True
            ),
            html.Div(id='hidden-div', style={'display': 'none'})
        ], className="bg-light p-4", style={"width": "20%"}),

        html.Div([
            dcc.Tabs(id='tabs', children=[
                dcc.Tab(label='Home', children=[
                    html.Div(id='summary-stats', className="d-flex justify-content-around"),
                    html.Div([
                        dcc.Graph(id='viewership-visits-bar', style={"width": "65%"}),
                        dcc.Graph(id='views-gauge', style={"width": "35%"}),
                    ], className="d-flex justify-content-around"),
                ], className="bg-light", style={"padding": "10px"}),

                dcc.Tab(label='Viewership Statistics', children=[
                    html.Div([
                        html.Div(dcc.Graph(id='sport-preferences'), style={"width": "50%"}),
                        html.Div(dcc.Graph(id='conversion-rate'), style={"width": "50%"}),
                    ], className="d-flex justify-content-around"),
                ], className="bg-light", style={"padding": "10px"}),

                dcc.Tab(label='Concurrent Sports Events', children=[
                    dcc.Graph(id='viewership-stats'),
                ], className="bg-light", style={"padding": "10px"}),

                dcc.Tab(label='Geographical Locations', children=[
                    dcc.Graph(id='geo-distribution'),
                ], className="bg-light", style={"padding": "10px"}),

                dcc.Tab(label='Viewership Over Time', children=[
                    dcc.Graph(id='viewership-times'),
                ], className="bg-light", style={"padding": "10px"}),
            ]),
        ], style={"width": "75%"}),

    ], className="d-flex justify-content-center"),
], className="bg-white")

@app.callback(
    Output('country-dropdown', 'options'),
    Input('continent-dropdown', 'value')
)
def set_country_options(selected_continents):
    if not selected_continents:
        return []
    countries = [country for continent in selected_continents for country in countries_by_continent.get(continent, [])]
    return [{'label': country, 'value': country} for country in countries]

@app.callback(
    Output('hidden-div', 'children'),
    Input('continent-dropdown', 'value'),
    Input('country-dropdown', 'value'),
    Input('sport-dropdown', 'value'),
    Input('age-dropdown', 'value'),
    Input('gender-dropdown', 'value'),
    Input('tabs', 'value')
)
def save_filter_options(continent_value, country_value, sport_value, age_value, gender_value, tab):
    return [
        html.Div(id=f'{tab}-continent', children=continent_value),
        html.Div(id=f'{tab}-country', children=country_value),
        html.Div(id=f'{tab}-sport', children=sport_value),
        html.Div(id=f'{tab}-age', children=age_value),
        html.Div(id=f'{tab}-gender', children=gender_value)
    ]

@app.callback(
    Output('viewership-stats', 'figure'),
    Input('continent-dropdown', 'value'),
    Input('sport-dropdown', 'value'),
    Input('age-dropdown', 'value'),
    Input('gender-dropdown', 'value')
)
def update_viewership_stats(continents, sports, ages, genders):
    if not continents:
        continents = ['Asia']  # Default continent

    filtered_df = df[df['Continent'].isin(continents)]
    if sports:
        filtered_df = filtered_df[filtered_df['Event'].isin(sports)]
    if ages:
        filtered_df = filtered_df[filtered_df['Age'].isin(ages)]
    if genders:
        filtered_df = filtered_df[filtered_df['Gender'].isin(genders)]

    fig = px.area(filtered_df, x='Time', y='Views', color='Event', title=f'Viewership Statistics for {", ".join(continents)}')
    fig.update_layout(template='plotly_white', xaxis_title='Time', yaxis_title='Views')

    return fig

@app.callback(
    Output('summary-stats', 'children'),
    Input('continent-dropdown', 'value'),
    Input('sport-dropdown', 'value'),
    Input('age-dropdown', 'value'),
    Input('gender-dropdown', 'value')
)
def update_summary_stats(continents, sports, ages, genders):
    if not continents:
        continents = ['Asia']  # Default continent

    filtered_df = df[df['Continent'].isin(continents)]
    if sports:
        filtered_df = filtered_df[filtered_df['Event'].isin(sports)]
    if ages:
        filtered_df = filtered_df[filtered_df['Age'].isin(ages)]
    if genders:
        filtered_df = filtered_df[filtered_df['Gender'].isin(genders)]

    total_viewers = filtered_df['Views'].sum()
    total_visits = filtered_df['Visits'].sum()
    avg_views = filtered_df['Views'].mean()
    avg_visits = filtered_df['Visits'].mean()

    return [
        html.Div([
            html.H5("Total Viewers", className="card-title"),
            html.P(f"{total_viewers/1000:.2f} M", className="card-text")
        ], className="card bg-primary text-white text-center p-3"),
        html.Div([
            html.H5("Total Visits", className="card-title"),
            html.P(f"{total_visits/1000:.2f} M", className="card-text")
        ], className="card bg-success text-white text-center p-3"),
        html.Div([
            html.H5("Average Views", className="card-title"),
            html.P(f"{avg_views:.2f}", className="card-text")
        ], className="card bg-info text-white text-center p-3"),
        html.Div([
            html.H5("Average Visits", className="card-title"),
            html.P(f"{avg_visits:.2f}", className="card-text")
        ], className="card bg-warning text-white text-center p-3")
    ]

@app.callback(
    Output('viewership-visits-bar', 'figure'),
    Input('continent-dropdown', 'value'),
    Input('sport-dropdown', 'value'),
    Input('age-dropdown', 'value'),
    Input('gender-dropdown', 'value')
)
def update_viewership_visits_bar(continents, sports, ages, genders):
    if not continents:
        continents = ['Asia']  # Default continent

    filtered_df = df[df['Continent'].isin(continents)]
    if sports:
        filtered_df = filtered_df[filtered_df['Event'].isin(sports)]
    if ages:
        filtered_df = filtered_df[filtered_df['Age'].isin(ages)]
    if genders:
        filtered_df = filtered_df[filtered_df['Gender'].isin(genders)]

    aggregated_df = filtered_df.groupby('Event').sum().reset_index()

    title = f"Average Viewership and Visits by Event for {', '.join(continents)}"
    if sports:
        title += f" - Sports: {', '.join(sports)}"
    if ages:
        title += f" - Ages: {', '.join(ages)}"
    if genders:
        title += f" - Genders: {', '.join(genders)}"

    fig = px.bar(aggregated_df, x='Event', y=['Views', 'Visits'], barmode='group', title=title)
    fig.update_layout(template='plotly_white', xaxis_title='Event', yaxis_title='Average (in thousands)')

    return fig



@app.callback(
    Output('views-gauge', 'figure'),
    Input('continent-dropdown', 'value'),
    Input('sport-dropdown', 'value'),
    Input('age-dropdown', 'value'),
    Input('gender-dropdown', 'value')
)
def update_views_gauge(continents, sports, ages, genders):
    if not continents:
        continents = ['Asia']  # Default continent

    filtered_df = df[df['Continent'].isin(continents)]
    if sports:
        filtered_df = filtered_df[filtered_df['Event'].isin(sports)]
    if ages:
        filtered_df = filtered_df[filtered_df['Age'].isin(ages)]
    if genders:
        filtered_df = filtered_df[filtered_df['Gender'].isin(genders)]

    total_views = filtered_df['Views'].sum()

    title = f"Total Views for {', '.join(continents)}"
    if sports:
        title += f" - Sports: {', '.join(sports)}"
    if ages:
        title += f" - Ages: {', '.join(ages)}"
    if genders:
        title += f" - Genders: {', '.join(genders)}"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=total_views / 1000,
        title={'text': title},
        gauge={'axis': {'range': [None, 2000]}}
    ))

    fig.update_layout(template='plotly_white')

    return fig


@app.callback(
    Output('sport-preferences', 'figure'),
    Input('continent-dropdown', 'value'),
    Input('country-dropdown', 'value'),
    Input('age-dropdown', 'value'),
    Input('gender-dropdown', 'value')
)
def update_sport_preferences(continents, countries, ages, genders):
    if not continents:
        continents = ['Asia']  # Default continent

    filtered_df = df[df['Continent'].isin(continents)]
    if countries:
        filtered_df = filtered_df[filtered_df['Country'].isin(countries)]
    if ages:
        filtered_df = filtered_df[filtered_df['Age'].isin(ages)]
    if genders:
        filtered_df = filtered_df[filtered_df['Gender'].isin(genders)]

    sport_counts = filtered_df['Event'].value_counts().reset_index()
    sport_counts.columns = ['Event', 'Count']

    title = f"Sport Preferences for {', '.join(continents)}"
    if countries:
        title += f" - Countries: {', '.join(countries)}"
    if ages:
        title += f" - Ages: {', '.join(ages)}"
    if genders:
        title += f" - Genders: {', '.join(genders)}"

    fig = px.pie(sport_counts, names='Event', values='Count', title=title)
    fig.update_layout(template='plotly_white')

    return fig


@app.callback(
    Output('conversion-rate', 'figure'),
    Input('continent-dropdown', 'value'),
    Input('country-dropdown', 'value'),
    Input('age-dropdown', 'value'),
    Input('gender-dropdown', 'value')
)
def update_conversion_rate(continents, countries, ages, genders):
    if not continents:
        continents = ['Asia']  # Default continent

    filtered_df = df[df['Continent'].isin(continents)]
    if countries:
        filtered_df = filtered_df[filtered_df['Country'].isin(countries)]
    if ages:
        filtered_df = filtered_df[filtered_df['Age'].isin(ages)]
    if genders:
        filtered_df = filtered_df[filtered_df['Gender'].isin(genders)]

    filtered_df['Conversion Rate'] = filtered_df['Views'] / filtered_df['Visits']

    conversion_rate_df = filtered_df.groupby('Event')['Conversion Rate'].mean().reset_index()

    title = f"Conversion Rate by Sport for {', '.join(continents)}"
    if countries:
        title += f" - Countries: {', '.join(countries)}"
    if ages:
        title += f" - Ages: {', '.join(ages)}"
    if genders:
        title += f" - Genders: {', '.join(genders)}"

    fig = px.bar(conversion_rate_df, x='Event', y='Conversion Rate', title=title)
    fig.update_layout(template='plotly_white', xaxis_title='Event', yaxis_title='Conversion Rate')

    return fig


@app.callback(
    Output('geo-distribution', 'figure'),
    Input('continent-dropdown', 'value'),
    Input('sport-dropdown', 'value'),
    Input('age-dropdown', 'value'),
    Input('gender-dropdown', 'value')
)
def update_geo_distribution(continents, sports, ages, genders):
    if not continents:
        continents = ['Asia']  # Default continent

    filtered_df = df[df['Continent'].isin(continents)]
    if sports:
        filtered_df = filtered_df[filtered_df['Event'].isin(sports)]
    if ages:
        filtered_df = filtered_df[filtered_df['Age'].isin(ages)]
    if genders:
        filtered_df = filtered_df[filtered_df['Gender'].isin(genders)]

    geo_df = filtered_df.groupby('Country')['Views'].sum().reset_index()

    title = f"Viewership by Country for {', '.join(continents)}"
    if sports:
        title += f" - Sports: {', '.join(sports)}"
    if ages:
        title += f" - Ages: {', '.join(ages)}"
    if genders:
        title += f" - Genders: {', '.join(genders)}"

    fig = px.choropleth(geo_df, locations='Country', locationmode='country names', color='Views', title=title)
    fig.update_layout(template='plotly_white')

    return fig


@app.callback(
    Output('viewership-times', 'figure'),
    Input('continent-dropdown', 'value'),
    Input('sport-dropdown', 'value'),
    Input('age-dropdown', 'value'),
    Input('gender-dropdown', 'value')
)
def update_viewership_times(continents, sports, ages, genders):
    if not continents:
        continents = ['Asia']  # Default continent

    filtered_df = df[df['Continent'].isin(continents)]
    if sports:
        filtered_df = filtered_df[filtered_df['Event'].isin(sports)]
    if ages:
        filtered_df = filtered_df[filtered_df['Age'].isin(ages)]
    if genders:
        filtered_df = filtered_df[filtered_df['Gender'].isin(genders)]

    time_df = filtered_df.groupby('Time')['Views'].sum().reset_index()

    title = f"Viewership Over Time for {', '.join(continents)}"
    if sports:
        title += f" - Sports: {', '.join(sports)}"
    if ages:
        title += f" - Ages: {', '.join(ages)}"
    if genders:
        title += f" - Genders: {', '.join(genders)}"

    fig = px.line(time_df, x='Time', y='Views', title=title)
    fig.update_layout(template='plotly_white', xaxis_title='Time', yaxis_title='Views')

    return fig


if __name__ == '__main__':
    app.run_server(debug=True)
