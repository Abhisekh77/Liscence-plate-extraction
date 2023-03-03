from yolov5_tflite_webcam_inference import detect_video
import os
import psycopg2
import jinja2

# create a Jinja2 environment
env = jinja2.Environment(loader=jinja2.FileSystemLoader('.'))

# load the template file
template = env.get_template('index.html')

if __name__ == "__main__":
    
    # Connect to the database
    conn = psycopg2.connect(
    host='localhost',
    database='platenumber',
    user='postgres',
    password='2256',
    port='5432'
    )

    # Open a cursor to perform database operations
    cur = conn.cursor()

    # Create the "output" table if it doesn't exist
    cur.execute("""
        CREATE TABLE IF NOT EXISTS platedetails (
            id SERIAL PRIMARY KEY,
            value TEXT NOT NULL,
            fine_amount numeric DEFAULT 500  
        )
    """)
    
    BASE_PATH = os.getcwd()
    print(BASE_PATH)
    # detect plate
    value = detect_video(weights=BASE_PATH+"/models/custom_plate.tflite", labels=BASE_PATH+"/labels/plate.txt", conf_thres=0.25, iou_thres=0.45,
                         img_size=640, webcam=0)
    print("inside main", value)
    # detect_image(weights="./models/character.tflite", labels="./labels/number.txt", conf_thres=0.25, iou_thres=0.45,
    #              image_url="./output/cropped/cropped1.jpg", img_size=640)
    # detect character
    print("abhi")
    
    # Insert the output value into the "output" table
    cur.execute("INSERT INTO platedetails (value) VALUES (%s)", (str(value),))
   #INSERT INTO table_name (column_name) VALUES (integer_value);
    
    print("Hello abhi")
    # Retrieve all the rows from the "platedetail" table
    a=cur.execute("SELECT * FROM platedetails")
    # Print the retrieved rows to the console
    rows = cur.fetchall()
    for row in rows:
       print("row",row)
    print("rows",rows)
    table_html = template.render(rows=rows)
     
    # Commit the transaction
    conn.commit()

    # Write the HTML to a file
    with open('output.html', 'w') as f:
        f.write(table_html)

    # Close the cursor and connection
    cur.close()
    conn.close()
    
