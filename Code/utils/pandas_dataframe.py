from IPython.display import display, HTML
import pandas as pd

# let data frame display flexible, show whole value in cell
def grid_display(list_df, list_df_name, list_number_of_data, row = 1, col=1, fill = 'col'):
    """how to use:
    grid_display(list_df = [df_1, df_2, df_3, df_4], 
                list_df_name = ['name_1', 'name_2', 'name_3', 'name_4'],
                list_number_of_data = [5, 28, 6, 5], row = 2, col = 2, fill = 'col')
    
    """
    
    
    html_table = "<table style='width:100%; border:0px'>{content}</table>"
    html_row = "<tr style='border:0px'>{content}</tr>"
    html_cell = "<td style='width:{width}%;vertical-align:top;border:0px'>{{content}}</td>"
    html_cell = html_cell.format(width=100/col)
    
    li = []
    for i in range(len(list_df)):
        if isinstance(list_df[i], pd.DataFrame):
            li.append(list_df[i].head(list_number_of_data[i]).
                      style.set_table_attributes("style='display:inline'").
                      set_caption(f'<b><H2>{list_df_name[i]}<H2></b>')
                     )
        else:
            li.append(list_df[i]
                      .set_caption(f'<b><H2>{list_df_name[i]}<H2></b>')
                     )
    
    
    cell = [ html_cell.format(content=df.to_html()) for df in li[:row*col] ]
    cell += col * [html_cell.format(content="")] # pad

    if fill == 'row': #fill in rows first (first row: 0,1,2,... col-1)
        grid = [ html_row.format(content="".join(cell[i:i+col])) for i in range(0,row*col,col)]

    if fill == 'col': #fill columns first (first column: 0,1,2,..., rows-1)
        grid = [ html_row.format(content="".join(cell[i:row*col:row])) for i in range(0,row)]
    
    display(HTML(html_table.format(content="".join(grid))))