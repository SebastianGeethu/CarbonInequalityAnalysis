import dash
import geopandas as gpd
import json
import os
import plotly.express as px
from dash.dependencies import Input, Output, State
from dash import html, dcc
from dash import dash_table as dt
from functools import reduce
import pandas as pd
from turfpy.measurement import bbox
import plotly.graph_objects as go

df = pd.read_csv('CombinedDoc.csv')
df['Year'] = pd.to_numeric(df['Year'])
df['Per Capita Income'] = pd.to_numeric(df['Per Capita Income'])

# For ploting the first line graph to display income vs year with country and la selection
la_sorted = sorted(df[df['Country'] == sorted(df['Country'].unique())[0]]['Local Authority'].unique())
default_la = [la_sorted[0]]
# Specify data types for each column
column_types = {
    "Year": "numeric",
    "Country": "text",
    "Region": "text",
    "Local Authority Code": "text",
    "Local Authority": "text",
    'Per Capita Income': "numeric",
    "Per Capita Electricity": "numeric",
    "Per Capita Gas": "numeric",
    "Per Capita Other": "numeric",
    "Per Capita Total": "numeric",
    "Income Group": "text",
}
# Create columns list with data types
columns = [{"name": i, "id": i, "type": column_types.get(i, "text")} for i in df.columns]


# Defining functions
def read_lad_geojson(country):
    country_jsonfile = country + "_LAD_Boundaries.json"

    if os.path.exists(country_jsonfile):
        with open(country_jsonfile) as f:
            census_lads = json.load(f)
    else:
        lad_gdf = gpd.read_file("LAD_DEC_2022_UK_BFC_V2.shp")

        # Simplify geometry
        lad_gdf.geometry = lad_gdf.geometry.simplify(0.001, preserve_topology=True)

        # Select necessary columns
        lad_gdf = lad_gdf[['LAD22CD', 'geometry']]
        # simplify geometry to 1000m accuracy
        lad_gdf["geometry"] = (
            lad_gdf.to_crs(lad_gdf.estimate_utm_crs()).simplify(800).to_crs(lad_gdf.crs))
        lad_gdf.set_index("LAD22CD")
        lad_gdf.to_crs(epsg=4326, inplace=True)

        # Serialize GeoDataFrame to GeoJSON format in-memory
        lad_json_data = lad_gdf.to_json()

        # Deserialize GeoJSON data back to Python objects
        census_lads = json.loads(lad_json_data)

        # Filter 'features' based on 'LAD22CD'
        census_lads['features'] = [f for f in census_lads['features'] if
                                   f['properties']['LAD22CD'].startswith(country[0])]

        # Save the resulting GeoJSON directly to country_jsonfile
        with open(country_jsonfile, 'w') as f:
            json.dump(census_lads, f)

    return census_lads


def get_max_value(year, country):
    return df[(df['Year'] == year) & (df['Country'] == country)]['Per Capita Income'].max()


def get_min_value(year, country):
    return df[(df['Year'] == year) & (df['Country'] == country)]['Per Capita Income'].min()


def get_max_value_emission(year, country, emission):
    # print("emission = ", emission)
    return df[(df['Year'] == year) & (df['Country'] == country)][emission].max()


def get_min_value_emission(year, country, emission):
    # print("emission = ", emission)
    return df[(df['Year'] == year) & (df['Country'] == country)][emission].min()


df_dict = {year: df[df["Year"] == year] for year in range(2015, 2022)}
countries = df["Country"].unique().tolist()

lad_ids = [list(map(lambda f: f['properties']['LAD22CD'], read_lad_geojson(country)["features"])) for country in
           countries]


def blank_fig():
    # Blank figure for initial Dash display
    fig = go.Figure(go.Scatter(x=[], y=[]))
    fig.update_layout(template=None)
    fig.update_xaxes(showgrid=False, showticklabels=False, zeroline=False)
    fig.update_yaxes(showgrid=False, showticklabels=False, zeroline=False)
    return fig


app = dash.Dash(__name__, meta_tags=[{"name": "viewport", "content": "width=device-width"}])
app.title= "Carbon Inequality Analysis Dashboard"
app.layout = html.Div((
    html.Div([
        html.Div([
            html.H1('Carbon Inequality Analysis Dashboard',
                    style={'margin-bottom': '0px', 'font-weight': 'bold', 'color': 'white'}),
        ])
    ], className="column", id="title1"),
    html.Div([], style={'height': '50px'}),

    html.Div([

        html.Div([
            html.P('Year', className='fix_label', style={'color': 'white'}),
            dcc.Slider(id='year',
                       included=False,
                       updatemode='drag',
                       tooltip={'always_visible': True},
                       min=2015,
                       max=2021,
                       step=1,
                       value=2021,
                       marks={str(yr): str(yr) for yr in range(2015, 2022)},
                       className='dcc_compon'),

        ], className="one-half column", id="title2"),

        html.Div([
            dcc.RadioItems(id='y-axis-selector',
                           labelStyle={"display": "inline-block"},
                           value='Per Capita Total',
                           options=[{'label': 'Gas Emission', 'value': 'Per Capita Gas'},
                                    {'label': 'Electricity Emission', 'value': 'Per Capita Electricity'},
                                    {'label': 'Other Emission', 'value': 'Per Capita Other'},
                                    {'label': 'Total Emission', 'value': 'Per Capita Total'}, ],
                           style={'text-align': 'center', 'color': 'white'}, className='dcc_compon'),
        ], className="one-half column", id='title3'),

    ], id="header", className="row flex-display", style={"margin-bottom": "25px"}),

    html.Div([
        html.Div([
            dcc.Dropdown(
                id='country_choice',
                options=[{'label': cn, 'value': cn} for cn in sorted(df['Country'].unique())],
                value=sorted(df['Country'].unique())[0],
                multi=True,
                className='custom-dropdown'

            ),
            dcc.Dropdown(
                id='local-authority',
                options=[],
                value=default_la,
                multi=True,
                className='custom-dropdown'
            ),

            html.Div([dcc.Graph(id='income-vs-years-plot',
                                config={'displayModeBar': 'hover'}, style={'height': '350px'})],
                     style={'display': 'inline-block', 'width': '50%'}),
            html.Div([dcc.Graph(id='dynamic-plot',
                                config={'displayModeBar': 'hover'}, style={'height': '350px'}), ],
                     style={'display': 'inline-block', 'width': '50%'})

        ], className='create_container2 columns', style={'height': '500px'}),

    ], className="row flex-display"),

    html.Div((

    )),
    html.Div((
        # html.Div([
        #
        # ], className='create_container2 four columns', style={'height': '600px'}),

        html.Div([
            dcc.RadioItems(id='country',
                           labelStyle={"display": "inline-block"},
                           value='England',
                           options=[{'label': 'England', 'value': 'England'},
                                    {'label': 'Scotland', 'value': 'Scotland'},
                                    {'label': 'Wales', 'value': 'Wales'},
                                    {'label': 'Northern Ireland', 'value': 'Northern Ireland'}
                                    ],
                           style={'text-align': 'center', 'color': 'white'}, className='dcc_compon'),
            html.Div([dcc.Graph(id='map')], style={'display': 'inline-block', 'width': '50%'}),
            html.Div([dcc.Graph(id='map2')], style={'display': 'inline-block', 'width': '50%'}),
        ], className='create_container2 columns'),

    ), className="row flex-display"),
    html.Div((
        html.Div([
            dcc.Graph(id='emission-bar-chart')
        ], className='create_container2 four columns'),
        html.Div([
            dt.DataTable(id='datasettable',
                         sort_action="native",
                         sort_mode="multi",
                         virtualization=True,
                         style_cell={'textAlign': 'left',
                                     'min-width': '100px',
                                     'backgroundColor': '#1f2c56',
                                     'color': '#FEFEFE',
                                     'border-bottom': '0.01rem solid #19AAE1',
                                     },
                         style_as_list_view=True,
                         style_header={
                             'backgroundColor': '#1f2c56',
                             'fontWeight': 'bold',
                             'font': 'Lato, sans-serif',
                             'color': 'orange',
                             'border': '#1f2c56',
                         },
                         style_data={'textOverflow': 'hidden', 'color': 'white'},
                         fixed_rows={'headers': True},
                         ),
            html.P(
                "*The data presented herein has been sourced from the official United Kingdom government website and "
                "subjected to processing. "
                "Emission figures are denoted in metric tons of carbon dioxide equivalent (tCO2e), while income values are "
                "expressed in British Pound Sterling (£).",
                style={'font-family': 'sans-serif', 'color': 'white', 'font-size': '10px', 'font-style': 'italic',
                       'text-align': 'left'}
            ),

        ], className='create_container2 eight columns'),

    ), className="row flex-display")

), id="mainContainer", style={"display": "flex", "flex-direction": "column"})


@app.callback(
    Output('local-authority', 'options'),
    Input('country_choice', 'value')
)
def update_local_authorities(selected_countries):
    # Convert the input to a list if it's a string
    if isinstance(selected_countries, str):
        selected_countries = [selected_countries]
    # Handle case when no country is selected
    if not selected_countries:
        return []
    # Filter the DataFrame based on selected countries
    filtered_df = df[df['Country'].isin(selected_countries)]

    # Get unique local authorities from the filtered DataFrame
    local_authorities = sorted(filtered_df['Local Authority'].unique())

    # Create options for the Local Authority dropdown
    options = [{'label': la, 'value': la} for la in local_authorities]

    return options


@app.callback(

    Output('income-vs-years-plot', 'figure'),
    Input('local-authority', 'value')

)
def update_plots(selected_local_authorities):
    filtered_data = df[df['Local Authority'].isin(selected_local_authorities)]
    income_vs_years_plot = px.line(filtered_data, x='Year', y='Per Capita Income', color='Local Authority',
                                   title="Per Capita Income Trends Over Years")

    income_vs_years_plot.update_layout(
        plot_bgcolor='#1f2c56',
        paper_bgcolor='#1f2c56',
        titlefont={
            'color': 'white',
            'size': 15},
        xaxis=dict(title='<b>Year</b>',
                   title_font=dict(size=16),
                   color='orange',
                   showline=True,
                   showgrid=False,
                   gridwidth=0.1,
                   showticklabels=True,
                   linecolor='orange',
                   linewidth=0.5,
                   ticks='outside',
                   tickfont=dict(
                       family='Arial',
                       size=12,
                       color='orange'),
                   gridcolor='#456575'
                   ),

        yaxis=dict(title='<b>Income(£)</b>',
                   title_font=dict(size=16),
                   # autorange='reversed',
                   color='orange',
                   showline=True,
                   showgrid=True,
                   gridwidth=0.1,
                   showticklabels=True,
                   linecolor='orange',
                   linewidth=0.5,
                   ticks='outside',
                   tickfont=dict(
                       family='Arial',
                       size=12,
                       color='orange'),
                   gridcolor='#456575'

                   ),

        legend={
            'font': {
                'size': 10  # Set the font size as needed
            },
            'bgcolor': '#1f2c56',
        },

        font=dict(
            family="Arial",
            size=10,
            color='white'),
    )

    return income_vs_years_plot


@app.callback(
    Output('datasettable', 'data'),
    [Input('year', 'value'),
     Input('y-axis-selector', 'value')]
)
def display_table(select_year, emission):
    data_table = df[(df['Year'] == select_year)]
    columns = ['Year', 'Local Authority Code', 'Local Authority', 'Region', 'Country',
               'Per Capita Income', emission, 'Income Group']
    data_table = data_table[columns]
    return data_table.to_dict('records')


@app.callback(
    Output('dynamic-plot', 'figure'),
    [Input('local-authority', 'value'),
     Input('y-axis-selector', 'value')]
)
def update_dynamic_plot(selected_local_authorities, selected_y_axis):
    if not selected_local_authorities:
        empty_plot = px.line()
        empty_plot.update_layout(
            title='Household Emission Trends Over Years',
            plot_bgcolor='#1f2c56',
            paper_bgcolor='#1f2c56',
            titlefont={
                'color': 'white',
                'size': 15},
            xaxis=dict(title='<b>Year</b>',
                       color='orange',
                       showline=True,
                       showgrid=False,
                       gridwidth=0.1,
                       showticklabels=True,
                       linecolor='orange',
                       linewidth=0.5,
                       ticks='outside',
                       tickfont=dict(
                           family='Arial',
                           size=12,
                           color='orange'),
                       gridcolor='#456575'
                       ),

            yaxis=dict(title='<b></b>',
                       # autorange='reversed',
                       color='orange',
                       showline=True,
                       showgrid=True,
                       gridwidth=0.1,
                       showticklabels=True,
                       linecolor='orange',
                       linewidth=0.5,
                       ticks='outside',
                       tickfont=dict(
                           family='Arial',
                           size=12,
                           color='orange'),
                       gridcolor='#456575'

                       ),

            legend={
                'font': {
                    'size': 10  # Set the font size as needed
                },
                'bgcolor': '#1f2c56',
            },

            font=dict(
                family="Arial",
                size=10,
                color='white'),
        )
        return empty_plot
    filtered_data = df[df['Local Authority'].isin(selected_local_authorities)]
    # print(filtered_data)
    # Find the label for the selected_y_axis value in the options list
    # selected_y_label = next(option['label'] for option in y_axis_options if option['value'] == selected_y_axis)

    dynamic_plot = px.line(filtered_data, x='Year', y=selected_y_axis, color='Local Authority',
                           title=f'{selected_y_axis} Emission Trends Over Years')
    dynamic_plot.update_layout(
        yaxis_title=f'{selected_y_axis} Emission (tCO2e)',  # Use the label instead of the value
        # yaxis=dict(range=[0, dynamic_plot.data[0].y.max() * 1.1])
        plot_bgcolor='#1f2c56',
        paper_bgcolor='#1f2c56',
        titlefont={
            'color': 'white',
            'size': 15},
        xaxis=dict(title='<b>Year</b>',
                   color='orange',
                   showline=True,
                   showgrid=False,
                   gridwidth=0.1,
                   showticklabels=True,
                   linecolor='orange',
                   linewidth=0.5,
                   ticks='outside',
                   tickfont=dict(
                       family='Arial',
                       size=12,
                       color='orange'),
                   gridcolor='#456575'
                   ),

        yaxis=dict(title=f'<b>{selected_y_axis}(tCO2e)</b>',
                   # autorange='reversed',
                   color='orange',
                   showline=True,
                   showgrid=True,
                   gridwidth=0.1,
                   showticklabels=True,
                   linecolor='orange',
                   linewidth=0.5,
                   ticks='outside',
                   tickfont=dict(
                       family='Arial',
                       size=12,
                       color='orange'),
                   gridcolor='#456575'

                   ),

        legend={
            'font': {
                'size': 10  # Set the font size as needed
            },
            'bgcolor': '#1f2c56',
        },

        font=dict(
            family="Arial",
            size=10,
            color='white'),
    )
    # print("dynamic_plot", dynamic_plot)
    return dynamic_plot


@app.callback(
    Output('map', 'figure'),

    [Input('year', 'value'),
     Input('country', 'value')]
)
def update_graph_and_local_authorities(year, country):
    lad_df = df_dict[year]
    lad_max_value = get_max_value(year, country)
    lad_min_value = get_min_value(year, country)
    gj = read_lad_geojson(country)
    gj_bbox = reduce(lambda b1, b2: [min(b1[0], b2[0]), min(b1[1], b2[1]),
                                     max(b1[2], b2[2]), max(b1[3], b2[3])],
                     map(lambda f: bbox(f['geometry']), gj['features']))

    fig = px.choropleth(lad_df,
                        geojson=gj,
                        locations="Local Authority Code",
                        color="Per Capita Income",
                        color_continuous_scale=['#CCFFCC', '#3399FF', '#000066'],
                        # color_continuous_scale='solar',
                        # color_continuous_scale=[ '#a3dbd4','#6a9da1','#476983','#3a5260','#0083B8'],
                        range_color=(lad_min_value, lad_max_value),
                        featureidkey="properties.LAD22CD",
                        scope='europe',
                        hover_data=["Local Authority", "Income Group"],
                        title=f"Per Capita Income by Local Authority in {country} ({year})",

                        # projection=""
                        )

    fig.update_geos(
        center_lon=(gj_bbox[0] + gj_bbox[2]) / 2.0,
        center_lat=(gj_bbox[1] + gj_bbox[3]) / 2.0,
        lonaxis_range=[gj_bbox[0], gj_bbox[2]],
        lataxis_range=[gj_bbox[1], gj_bbox[3]],
        visible=False
    )

    # fig.update_traces(hoverinfo="location+z")
    fig.update_traces(
        hovertemplate="<b>%{customdata[0]}</b><br>" +
                      "Income: £%{z:.2f}<br>" +
                      "Income Group: %{customdata[1]}<extra></extra>",
        selector=dict(type="choropleth"))

    # fig.update_layout(margin=dict(l=0, r=0, b=0, t=30),
    #                   title_x=0.5,
    #                   width=1200, height=600)

    fig.update_layout(
        margin=dict(l=0, r=0, b=0, t=60),
        plot_bgcolor='#1f2c56',
        paper_bgcolor='#1f2c56',
        geo_bgcolor='#1f2c56',
        titlefont={
            'color': 'white',
            'size': 15},
        title_x=0.5,  # Center the title
        title_y=0.92,
        font=dict(
            family="sans-serif",
            size=10,
            color='white')
    )

    return fig


@app.callback(
    Output('map2', 'figure'),

    [Input('year', 'value'),
     Input('country', 'value'),
     Input('y-axis-selector', 'value')]
)
def update_graph_and_local_authorities(year, country, selected_y_axis):
    lad_df = df_dict[year]
    lad_max_value_emission = get_max_value_emission(year, country, selected_y_axis)
    lad_min_value_emission = get_min_value_emission(year, country, selected_y_axis)
    gj = read_lad_geojson(country)
    gj_bbox = reduce(lambda b1, b2: [min(b1[0], b2[0]), min(b1[1], b2[1]),
                                     max(b1[2], b2[2]), max(b1[3], b2[3])],
                     map(lambda f: bbox(f['geometry']), gj['features']))

    fig = px.choropleth(lad_df,
                        geojson=gj,
                        locations="Local Authority Code",
                        color=selected_y_axis,
                        color_continuous_scale=['#CCFFCC', '#3399FF', '#000066'],
                        range_color=(lad_min_value_emission, lad_max_value_emission),
                        featureidkey="properties.LAD22CD",
                        scope='europe',
                        hover_data=["Local Authority", "Income Group"],
                        title=f"{selected_y_axis} Emission(tCO2e) by Local Authority in {country} ({year})",
                        # projection=""
                        )

    fig.update_geos(
        center_lon=(gj_bbox[0] + gj_bbox[2]) / 2.0,
        center_lat=(gj_bbox[1] + gj_bbox[3]) / 2.0,
        lonaxis_range=[gj_bbox[0], gj_bbox[2]],
        lataxis_range=[gj_bbox[1], gj_bbox[3]],
        visible=False
    )

    # fig.update_traces(hoverinfo="location+z")
    fig.update_traces(hovertemplate="<b>%{customdata[0]}</b><br>" +
                                    "Total Per Capita Emission (tCO2e): %{z:.2f}<br>" +
                                    "Income Group: %{customdata[1]}<extra></extra>",
                      selector=dict(type="choropleth"))

    # fig.update_layout(margin=dict(l=0, r=0, b=0, t=30),
    #                   title_x=0.5,
    #                   width=1200, height=600)

    fig.update_layout(
        margin=dict(l=0, r=0, b=0, t=60),
        plot_bgcolor='#1f2c56',
        paper_bgcolor='#1f2c56',
        geo_bgcolor='#1f2c56',
        titlefont={
            'color': 'white',
            'size': 15},
        title_x=0.5,  # Center the title
        title_y=0.92,
        font=dict(
            family="sans-serif",
            size=10,
            color='white'),

    )

    return fig


@app.callback(
    Output('emission-bar-chart', 'figure'),
    [Input('y-axis-selector', 'value'),
     Input('year', 'value')]
)
def update_graph(selected_emission, selected_year):
    filtered_df = df[df['Year'] == selected_year]

    # Calculate mean for each income group
    grouped_df = filtered_df.groupby('Income Group', as_index=False)[selected_emission].mean().round(3)
    grouped_df_sorted = grouped_df.sort_values(by=selected_emission, ascending=False)
    fig = px.bar(
        grouped_df_sorted,
        y='Income Group',
        x=selected_emission,
        labels={selected_emission: f'Mean {selected_emission}'},
        title=f'{selected_emission} Emission by income group in year {selected_year}',
        template='ggplot2',
        orientation='h',
        # category_orders={'Income Group': income_group_order},  # Specify the order
        color_discrete_sequence=["#0083B8"]
    )

    fig.update_traces(showlegend=False)
    fig.update_layout(
        plot_bgcolor='#1f2c56',
        paper_bgcolor='#1f2c56',
        title={
            'y': 0.92,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'},
        titlefont={
            'color': 'white',
            'size': 15},

        hovermode='closest',
        margin=dict(t=80, r=0),

        xaxis=dict(title='<b></b>',
                   color='orange',
                   showline=True,
                   showgrid=True,
                   gridwidth=0.1,
                   showticklabels=True,
                   linecolor='orange',
                   linewidth=1,
                   ticks='outside',
                   tickfont=dict(
                       family='Arial',
                       size=12,
                       color='orange'),
                   gridcolor='#456575'
                   ),

        yaxis=dict(title='<b></b>',
                   autorange='reversed',
                   color='orange',
                   showline=False,
                   showgrid=False,
                   gridwidth=0.1,
                   showticklabels=True,
                   linecolor='orange',
                   linewidth=1,
                   ticks='outside',
                   tickfont=dict(
                       family='Arial',
                       size=12,
                       color='orange'),
                   gridcolor='#456575'

                   ),

        font=dict(
            family="sans-serif",
            size=15,
            color='white'),

    )

    #

    return fig


# Callback to update modal content
@app.callback(
    Output('modal-content', 'children'),
    Input('info-icon', 'n_clicks')
)
def update_modal_content(n_clicks):
    if n_clicks is None:
        return ""

    content = """
    This is some additional information.
    You can format it using Markdown.
    """
    return content


# Callbacks to toggle modal open/close
@app.callback(
    Output('modal', 'is_open'),
    [Input('modal-link', 'n_clicks'), Input('close-modal-link', 'n_clicks')],
    [State('modal', 'is_open')]
)
def toggle_modal(n_open, n_close, is_open):
    if n_open or n_close:
        return not is_open
    return is_open


if __name__ == '__main__':
    app.run_server(debug=False, host="0.0.0.0", port=8080)
