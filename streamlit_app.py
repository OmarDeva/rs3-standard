import streamlit as st
import pandas as pd
import numpy as np
import requests
import time  # For tracking time spent
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import normalize
from google.cloud import storage
from google.cloud import firestore
from google.cloud.firestore import Increment
from google.oauth2 import service_account
from io import BytesIO
from streamlit_star_rating import st_star_rating  # For star rating widget
from PIL import Image
import logging

# ============================
# Setup Logging
# ============================
logging.basicConfig(
    filename='app.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s:%(message)s'
)

def log_image_error(product_id, error_message):
    try:
        error_ref = db.collection("image_errors").document(product_id)
        error_ref.set({
            "error_message": error_message,
            "timestamp": firestore.SERVER_TIMESTAMP
        })
        logging.error(f"Image load failed for {product_id}: {error_message}")
    except Exception as e:
        logging.error(f"Failed to log image error for {product_id}: {e}")

# ============================
# Firestore Initialization
# ============================
def init_firestore():
    try:
        service_account_info = st.secrets["firebase"]
        credentials = service_account.Credentials.from_service_account_info(service_account_info)
        db = firestore.Client(credentials=credentials)
        logging.info("Firestore initialized successfully.")
        return db
    except Exception as e:
        logging.error(f"Failed to initialize Firestore: {e}")
        st.error("Failed to initialize Firestore.")
        st.stop()

db = init_firestore()

# ============================
# GCS Client Setup
# ============================
def setup_gcs_client():
    try:
        service_account_info = st.secrets["connections"]["gcs"]
        client = storage.Client.from_service_account_info(service_account_info)
        logging.info("GCS client initialized successfully.")
        return client
    except Exception as e:
        logging.error(f"Failed to initialize GCS client: {e}")
        st.error("Failed to initialize GCS client.")
        st.stop()

# ============================
# Caching Large File Downloads
# ============================
@st.cache_resource
def download_blob_to_memory_cached(bucket_name, source_blob_name):
    try:
        client = setup_gcs_client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(source_blob_name)
        data = blob.download_as_bytes()
        logging.info(f"Downloaded {source_blob_name} from bucket {bucket_name}.")
        return data
    except Exception as e:
        logging.error(f"Failed to download {source_blob_name} from bucket {bucket_name}: {e}")
        return None

# ============================
# Initialize Session State Variables
# ============================
if 'view_count' not in st.session_state:
    st.session_state['view_count'] = {}

if 'start_time_product' not in st.session_state:
    st.session_state['start_time_product'] = {}

if 'cart' not in st.session_state:
    st.session_state['cart'] = {}  # Initialize as empty dictionary for session state

if 'remaining_balance' not in st.session_state:
    st.session_state['remaining_balance'] = 270  # Default balance

# Initialize feedback_messages list in session state
if 'feedback_messages' not in st.session_state:
    st.session_state['feedback_messages'] = []

if 'start_time' not in st.session_state:
    st.session_state['start_time'] = time.time()

# ============================
# User Input for Registration
# ============================
st.sidebar.header("User Information")
user_id = st.sidebar.text_input("Enter your User ID (e.g., Prolific ID) to register")

# Check for user_id
if not user_id:
    st.warning("Please enter your User ID to proceed.")
    st.stop()

# User inputs for age and sex
user_age = st.sidebar.number_input("Enter your age", min_value=18, max_value=100, value=25)
user_sex = st.sidebar.selectbox("Select your sex", ["Male", "Female"])

# ============================
# Define Product IDs for Men and Women
# ============================
men_product_ids = [
    "1636", "1637", "1653", "1806", "1831", "2219", "2477", "2817", "2826", "2827",
    "2963", "3150", "3160", "3168", "3307", "3556", "3558", "3798", "3996", "3997",
    "4184", "4343", "4577", "5468", "5654", "5899", "8984", "44785"
]

women_product_ids = [
    "4072", "4140", "4146", "4147", "4148", "4149", "4512", "4548", "4570", "4571",
    "4631", "4639", "4655", "4742", "4743", "6644", "12421", "12683", "13356",
    "13361", "13555", "14340", "15257", "15261", "15267", "39896", "40483", "53732"
]

# ============================
# Load and Filter Products
# ============================
bucket_name = 'rec_bucket_1'
fashion_data = download_blob_to_memory_cached(bucket_name, 'fashion.csv')

if fashion_data is not None:
    try:
        fashion_df = pd.read_csv(BytesIO(fashion_data))
        fashion_df["ProductId"] = fashion_df["ProductId"].astype(str)
        
        # **Ensure 'Price' is numeric**
        fashion_df["Price"] = pd.to_numeric(fashion_df["Price"], errors='coerce')
        # Drop rows where 'Price' is NaN after conversion
        initial_len = len(fashion_df)
        fashion_df = fashion_df.dropna(subset=['Price'])
        final_len = len(fashion_df)
        if initial_len != final_len:
            logging.warning(f"Dropped {initial_len - final_len} rows due to non-numeric 'Price'.")
            st.warning(f"Dropped {initial_len - final_len} products due to invalid price data.")
        
        logging.info("Fashion data loaded and cleaned successfully.")
    except Exception as e:
        logging.error(f"Failed to read or clean fashion.csv: {e}")
        st.error("Failed to load product data.")
        st.stop()
else:
    st.error("Failed to download fashion.csv.")
    st.stop()

def get_filtered_products(user_sex, fashion_df):
    if user_sex == "Male":
        product_ids = men_product_ids
    else:
        product_ids = women_product_ids
    filtered_df = fashion_df[fashion_df['ProductId'].isin(product_ids)]
    logging.info(f"Filtered products for {user_sex}: {len(filtered_df)} items.")
    return filtered_df

filtered_products_df = get_filtered_products(user_sex, fashion_df)

# ============================
# Data Validation Function
# ============================
def validate_product(product):
    try:
        assert 'Price' in product, "Price field is missing."
        assert isinstance(product['Price'], (int, float)), "Price must be a number."
        if isinstance(product['Price'], float) and np.isnan(product['Price']):
            raise AssertionError("Price cannot be NaN.")
        assert isinstance(product['ProductId'], str), "ProductId must be a string."
        assert isinstance(product['ProductTitle'], str), "ProductTitle must be a string."
        assert isinstance(product['ImageURL'], str), "ImageURL must be a string."
        assert isinstance(product.get('quantity', 1), int), "Quantity must be an integer."
        return True, ""
    except AssertionError as ae:
        return False, str(ae)

# ============================
# Initialize User in Firestore
# ============================
def initialize_user(db, user_id, user_age, user_sex):
    try:
        user_ref = db.collection("users").document(user_id)
        user_doc = user_ref.get()
        if not user_doc.exists:
            user_ref.set({
                "user_age": user_age,
                "user_sex": user_sex,
                "time_spent": 0,
                "remaining_balance": 270
            })
            st.success("New user registered with a $270 budget.")
            logging.info(f"New user registered: {user_id}")
        else:
            user_data = user_doc.to_dict()
            # Update session state with remaining_balance from Firestore
            st.session_state['remaining_balance'] = user_data.get('remaining_balance', 270)
            # Restore other session state variables if needed
            if 'time_spent' in user_data:
                st.session_state['time_spent'] = user_data['time_spent']
            st.success(f"Welcome back, {user_id}! Your session data has been restored.")
            logging.info(f"Existing user session restored: {user_id}")
        
        # Fetch cart items from 'cart' subcollection and populate session_state['cart']
        cart_ref = user_ref.collection("cart")
        cart_docs = cart_ref.stream()
        cart_items = {}
        for doc in cart_docs:
            cart_items[doc.id] = doc.to_dict()
        st.session_state['cart'] = cart_items
        logging.info(f"Cart items loaded for user {user_id}.")
        
    except Exception as e:
        logging.error(f"Failed to initialize user {user_id}: {e}")
        st.error("Failed to initialize user session.")
        st.stop()

initialize_user(db, user_id, user_age, user_sex)

# ============================
# Firestore Interaction Functions
# ============================
def increment_product_view_count(db, product_id, user_id):
    try:
        # Update total view_count
        product_ref = db.collection("product_stats").document(product_id)
        product_ref.set({}, merge=True)
        product_ref.update({"view_count": Increment(1)})
        logging.info(f"Incremented view_count for product_id: {product_id}")

        # Update user's view count
        user_view_ref = product_ref.collection("user_views").document(user_id)
        user_view_ref.set({
            "view_count": Increment(1),
            "last_viewed": firestore.SERVER_TIMESTAMP
        }, merge=True)
        logging.info(f"Incremented view_count for user_id: {user_id} on product_id: {product_id}")
    except Exception as e:
        logging.error(f"Failed to increment view_count for {product_id} and user {user_id}: {e}")

def increment_product_click_count(db, product_id, user_id):
    try:
        # Update total click_count
        product_ref = db.collection("product_stats").document(product_id)
        product_ref.set({}, merge=True)
        product_ref.update({"click_count": Increment(1)})
        logging.info(f"Incremented click_count for product_id: {product_id}")

        # Update user's click count
        user_click_ref = product_ref.collection("user_clicks").document(user_id)
        user_click_ref.set({
            "click_count": Increment(1),
            "last_clicked": firestore.SERVER_TIMESTAMP
        }, merge=True)
        logging.info(f"Incremented click_count for user_id: {user_id} on product_id: {product_id}")
    except Exception as e:
        logging.error(f"Failed to increment click_count for {product_id} and user {user_id}: {e}")

def increment_product_time_spent(db, product_id, user_id, time_spent_seconds):
    try:
        # Update total time_spent for the product
        product_ref = db.collection("product_stats").document(product_id)
        product_ref.set({}, merge=True)
        product_ref.update({"time_spent": Increment(time_spent_seconds)})
        logging.info(f"Incremented total time_spent by {time_spent_seconds} seconds for product_id: {product_id}")

        # Update user's time_spent in the subcollection
        user_time_ref = product_ref.collection("user_time_spent").document(user_id)
        user_time_ref.set({
            "time_spent": Increment(time_spent_seconds),
            "last_viewed": firestore.SERVER_TIMESTAMP
        }, merge=True)
        logging.info(f"Incremented time_spent by {time_spent_seconds} seconds for user_id: {user_id} on product_id: {product_id}")
    except Exception as e:
        logging.error(f"Failed to increment time_spent for product {product_id} and user {user_id}: {e}")

def save_feedback_to_firestore(db, product_id, product_title, user_id, rating):
    try:
        # Save purchase_count directly under the product document
        feedback_ref = db.collection("product_feedback").document(product_id)
        feedback_ref.set({}, merge=True)
        feedback_ref.update({"purchase_count": Increment(1)})

        # Store individual ratings in a subcollection with user_id as document ID
        rating_ref = feedback_ref.collection("ratings").document(user_id)
        rating_ref.set({
            "user_id": user_id,
            "rating": rating,
            "timestamp": firestore.SERVER_TIMESTAMP
        }, merge=True)

        logging.info(f"Feedback saved for product_id: {product_id} by user: {user_id}")

    except Exception as e:
        logging.error(f"Failed to save feedback for {product_id} by user {user_id}: {e}")
        st.error(f"Failed to save feedback for {product_title}.")


def save_purchase_information(db, cart_items):
    try:
        if not cart_items:
            st.error("Your cart is empty.")
            logging.error("Cart is empty, cannot process purchase.")
            return

        total_price = sum(float(item.get('Price', 0)) * item.get('quantity', 1) for item in cart_items.values())
        logging.info(f"Total price for purchase: {total_price}")

        remaining_balance = st.session_state['remaining_balance']
        if total_price > remaining_balance:
            st.error("Insufficient balance.")
            logging.error("User has insufficient balance for purchase.")
            return

        user_ref = db.collection("users").document(user_id)
        # Save cart items to 'purchases' collection under user
        purchase_id = f"purchase_{int(time.time())}"
        purchases_ref = user_ref.collection("purchases").document(purchase_id)
        # Prepare cart items for saving
        processed_cart = {}
        for key in cart_items:
            item = cart_items[key]
            if isinstance(item, pd.Series):
                item_dict = item.to_dict()
            else:
                item_dict = item
            processed_cart[key] = {
                "ProductId": item_dict['ProductId'],
                "ProductTitle": item_dict['ProductTitle'],
                "ImageURL": item_dict['ImageURL'],
                "Price": item_dict['Price'],
                "quantity": item_dict.get('quantity', 1)
            }
        purchases_ref.set({
            "items": processed_cart,
            "total_price": total_price,
            "timestamp": firestore.SERVER_TIMESTAMP
        })
        logging.info(f"Saved purchase information for user {user_id} with purchase ID {purchase_id}.")

        # Update user's remaining_balance
        user_ref.update({
            "remaining_balance": Increment(-total_price)
        })
        logging.info(f"Deducted ${total_price} from user {user_id}'s balance.")

        # Clear the cart subcollection in Firestore
        cart_ref = user_ref.collection("cart")
        for doc in cart_ref.stream():
            doc.reference.delete()
        logging.info(f"Cleared cart for user {user_id} after purchase.")

        # Update session state
        st.session_state['remaining_balance'] -= total_price
        st.session_state['cart'] = {}
        st.success("Purchase processed successfully.")
    except Exception as e:
        logging.error(f"Failed to save purchase information for user {user_id}: {e}")
        st.error("Failed to process your purchase.")

# ============================
# Similar Products Function
# ============================
def get_similar_products_cnn(product_id, num_results, fashion_df, user_sex):
    try:
        if user_sex == "Male":
            gender_filter = "Men"
        elif user_sex == "Female":
            gender_filter = "Women"
        else:
            st.error("Invalid gender selected!")
            return []

        if gender_filter == "Men":
            men_features_data = download_blob_to_memory_cached('rec_bucket_1', 'Men_ResNet_features.npy')
            men_product_ids_data = download_blob_to_memory_cached('rec_bucket_1', 'Men_ResNet_feature_product_ids.npy')
            extracted_features = np.load(BytesIO(men_features_data))
            Productids = np.load(BytesIO(men_product_ids_data))
        elif gender_filter == "Women":
            women_features_data = download_blob_to_memory_cached('rec_bucket_1', 'Women_ResNet_features.npy')
            women_product_ids_data = download_blob_to_memory_cached('rec_bucket_1', 'Women_ResNet_feature_product_ids.npy')
            extracted_features = np.load(BytesIO(women_features_data))
            Productids = np.load(BytesIO(women_product_ids_data))

        Productids = [str(pid) for pid in Productids]

        if product_id not in Productids:
            st.error(f"Product ID {product_id} not found in Product IDs!")
            return []

        doc_id = Productids.index(product_id)
        extracted_features = normalize(extracted_features)

        pairwise_dist = pairwise_distances(extracted_features, extracted_features[doc_id].reshape(1, -1), metric='euclidean')
        indices = np.argsort(pairwise_dist.flatten())[1:num_results+1]
        similar_product_ids = [Productids[i] for i in indices]
        similar_products = fashion_df[fashion_df['ProductId'].isin(similar_product_ids)]

        logging.info(f"Retrieved {len(similar_products)} similar products for product_id: {product_id}")
        return similar_products
    except Exception as e:
        logging.error(f"Failed to get similar products for {product_id}: {e}")
        st.error("Failed to retrieve similar products.")
        return []

# ============================
# Fetch and Cache Image Function
# ============================
@st.cache_data(show_spinner=False)
def fetch_and_cache_image(url):
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (compatible; StreamlitApp/1.0)'
        }
        response = requests.get(url, timeout=5, headers=headers)
        response.raise_for_status()
        logging.info(f"Fetched image from URL: {url}")
        return response.content
    except requests.exceptions.RequestException as e:
        logging.error(f"Image fetch failed for {url}: {e}")
        return None

def resize_image(image_bytes, width=150):
    try:
        image = Image.open(BytesIO(image_bytes))
        image = image.convert("RGB")
        image.thumbnail((width, width))
        buf = BytesIO()
        image.save(buf, format="JPEG")
        logging.info("Image resized successfully.")
        return buf.getvalue()
    except Exception as e:
        logging.error(f"Error resizing image: {e}")
        return None

# ============================
# Display Products Function
# ============================
def display_products(product_df):
    for i in range(0, len(product_df), 4):
        cols = st.columns(4)
        for j, col in enumerate(cols):
            if i + j < len(product_df):
                product = product_df.iloc[i + j]
                with col:
                    image_bytes = fetch_and_cache_image(product['ImageURL'])
                    if image_bytes:
                        resized_image = resize_image(image_bytes, width=150)
                        if resized_image:
                            st.image(resized_image, width=150)
                        else:
                            st.image("https://via.placeholder.com/150", width=150)
                            st.error(f"Failed to process image for {product['ProductTitle']}")
                            log_image_error(product['ProductId'], "Image resizing failed.")
                    else:
                        st.image("https://via.placeholder.com/150", width=150)
                        st.error(f"Failed to load image for {product['ProductTitle']}")
                        log_image_error(product['ProductId'], "Image fetch failed.")

                    st.markdown(f"**{product['ProductTitle']}**")
                    st.markdown(f"Price: ${product['Price']}")

                    view_product_key = f"view_product_{product['ProductId']}_{i}_{j}"
                    if st.button(f"View {product['ProductTitle']}", key=view_product_key):
                        st.session_state.clicked_product = product
                        increment_product_click_count(db, product['ProductId'], user_id)
                        st.rerun()
                        

# ============================
# Show Product Page Function
# ============================
def show_product_page(product):
    product_id = product['ProductId']
    if product_id not in st.session_state['start_time_product']:
        st.session_state['start_time_product'][product_id] = time.time()
        logging.info(f"Start time recorded for product_id: {product_id}")

    st.markdown(f"### {product['ProductTitle']}")
    st.markdown(f"**Price**: ${product.get('Price', 'N/A')}")

    image_bytes = fetch_and_cache_image(product['ImageURL'])
    if image_bytes:
        resized_image = resize_image(image_bytes, width=300)
        if resized_image:
            st.image(resized_image, width=300)
        else:
            st.image("https://via.placeholder.com/300", width=300)
            st.error(f"Failed to process image for {product['ProductTitle']}")
            log_image_error(product['ProductId'], "Image resizing failed.")
    else:
        st.image("https://via.placeholder.com/300", width=300)
        st.error(f"Failed to load image for {product['ProductTitle']}")
        log_image_error(product['ProductId'], "Image fetch failed.")

    add_to_cart_key = f"add_to_cart_{product['ProductId']}"
    if st.button("Add to Cart", key=add_to_cart_key):
        try:
            # Convert 'Price' to float and assign back to 'product'
            product_price = float(product.get('Price', 0))

            if np.isnan(product_price):
                raise ValueError("Price cannot be NaN.")

            product['Price'] = product_price

            # Log the type and value of 'Price' for debugging
            logging.info(f"Product Price Type: {type(product['Price'])}, Value: {product['Price']}")

            # Calculate current total
            current_total = sum(float(item.get('Price', 0)) * item.get('quantity', 1) for item in st.session_state['cart'].values())
            remaining_budget = st.session_state.get('remaining_balance', 270) - current_total

            product_key = product['ProductId']
            if product_key in st.session_state['cart']:
                st.warning(f"**{product['ProductTitle']}** is already in your cart.")
            elif product_price > remaining_budget:
                st.error(f"Adding this product exceeds your remaining balance of ${remaining_budget:.2f}.")
            else:
                product['quantity'] = 1
                st.session_state['cart'][product_key] = product
                st.success(f"**{product['ProductTitle']}** added to cart.")
                
                # Validate product data
                is_valid, validation_msg = validate_product(product)
                if not is_valid:
                    raise ValueError(f"Validation Error: {validation_msg}")
                
                # Log the product data before Firestore write
                logging.info(f"Product data to be written to Firestore: {product}")
                
                # Add item to Firestore 'cart' subcollection
                user_ref = db.collection("users").document(user_id)
                cart_ref = user_ref.collection("cart").document(product_key)
                cart_ref.set({
                    "ProductId": str(product['ProductId']),
                    "ProductTitle": str(product['ProductTitle']),
                    "ImageURL": str(product['ImageURL']),
                    "Price": float(product['Price']),
                    "quantity": int(product.get('quantity', 1))
                })
                logging.info(f"Added product_id: {product_key} to user {user_id}'s cart.")
        except Exception as e:
            logging.error(f"Failed to add product to cart: {e}")
            st.error(f"Failed to add product to cart: {e}")  # Display the specific error message

    if product_id not in st.session_state['view_count']:
        increment_product_view_count(db, product_id, user_id)
        st.session_state['view_count'][product_id] = True
        logging.info(f"View count incremented for product_id: {product_id}")

    st.markdown("### Similar Products")
    similar_products = get_similar_products_cnn(product['ProductId'], 8, fashion_df, user_sex)
    if similar_products.empty:
        st.write("No similar products found.")
    else:
        for i in range(0, len(similar_products), 4):
            cols = st.columns(4)
            for j, col in enumerate(cols):
                if i + j < len(similar_products):
                    row = similar_products.iloc[i + j]
                    with col:
                        image_bytes = fetch_and_cache_image(row['ImageURL'])
                        if image_bytes:
                            resized_image = resize_image(image_bytes, width=150)
                            if resized_image:
                                st.image(resized_image, width=150)
                            else:
                                st.image("https://via.placeholder.com/150", width=150)
                                st.error(f"Failed to process image for {row['ProductTitle']}")
                                log_image_error(row['ProductId'], "Image resizing failed.")
                        else:
                            st.image("https://via.placeholder.com/150", width=150)
                            st.error(f"Failed to load image for {row['ProductTitle']}")
                            log_image_error(row['ProductId'], "Image fetch failed.")

                        st.markdown(f"**{row['ProductTitle']}**")
                        view_similar_key = f"view_similar_{row['ProductId']}_{i}_{j}"
                        if st.button(f"View {row['ProductTitle']}", key=view_similar_key):
                            st.session_state.clicked_product = row
                            increment_product_click_count(db, row['ProductId'], user_id)
                            st.rerun()

    if st.button("Back to Home", key="back_to_home"):
        if 'clicked_product' in st.session_state:
            del st.session_state['clicked_product']
            logging.info("Navigated back to home page.")
            st.rerun()

    if product_id in st.session_state['start_time_product']:
        end_time = time.time()
        time_spent = end_time - st.session_state['start_time_product'][product_id]
        increment_product_time_spent(db, product_id, user_id, int(time_spent))
        logging.info(f"Time spent on product_id {product_id} by user {user_id}: {int(time_spent)} seconds")
        del st.session_state['start_time_product'][product_id]
        save_session_data()

# ============================
# Save Session Data Function
# ============================
def save_session_data():
    try:
        # Update remaining_balance based on session state
        # No need to handle cart here since it's managed as a subcollection
        session_data = {
            "user_age": user_age,
            "user_sex": user_sex,
            "time_spent": time.time() - st.session_state['start_time'],
            "remaining_balance": st.session_state.get('remaining_balance', 270)
        }
        db.collection("users").document(user_id).set(session_data, merge=True)
        logging.info(f"Session data saved for user_id: {user_id}")
    except Exception as e:
        logging.error(f"Failed to save session data for user {user_id}: {e}")
        st.error(f"Failed to save session data.")

# ============================
# Shopping Cart Display in Sidebar
# ============================
with st.sidebar:
    st.header("Shopping Cart")
    cart_items = st.session_state['cart']
    if cart_items:
        total_price = sum(float(item.get('Price', 0)) * item.get('quantity', 1) for item in cart_items.values())
        remaining_balance = st.session_state.get('remaining_balance', 270) - total_price
        st.write(f"**Total Price:** ${total_price:.2f}")
        st.write(f"**Remaining Balance:** ${remaining_balance:.2f}")

        progress = total_price / 270  # Assuming 270 is the total budget
        progress = min(progress, 1.0)
        st.progress(progress)

        if st.button("Proceed to Checkout", key="proceed_to_checkout"):
            st.session_state['checkout'] = True
            st.rerun()
            logging.info(f"Proceed to Checkout clicked by user {user_id}")
    else:
        st.write("Your cart is empty.")

# ============================
# Checkout Page
# ============================
if "checkout" in st.session_state and st.session_state['checkout']:
    st.title("Checkout")
    cart_items = st.session_state['cart']
    
    # Define cart_keys from the cart_items
    cart_keys = list(cart_items.keys())  # This gets the keys (product IDs) from the cart_items dictionary
    
    if cart_items:
        total_price = sum(float(item.get('Price', 0)) * item.get('quantity', 1) for item in cart_items.values())
        remaining_balance_after_purchase = st.session_state.get('remaining_balance', 270) - total_price
        st.markdown(f"**Total Price:** ${total_price:.2f}")
        st.markdown(f"**Remaining Balance After Purchase:** ${remaining_balance_after_purchase:.2f}")

        # Display cart items
        for idx, key in enumerate(cart_keys):
            item = cart_items[key]
            cols = st.columns([1, 2, 3])
            with cols[0]:
                st.image(item['ImageURL'], width=80)
            with cols[1]:
                st.markdown(f"**{item['ProductTitle']}**")
                st.markdown(f"**Price:** ${item.get('Price', 'N/A')}")
                st.markdown(f"**Quantity:** {item.get('quantity', 1)}")
                rating_key = f"rate_{item['ProductId']}_{idx}"
                st.markdown("**Rate this product:**")
                rating = st_star_rating("", maxValue=5, defaultValue=st.session_state.get(rating_key, 0), size=25, key=rating_key)
            with cols[2]:
                if st.button(f"Remove", key=f"remove_{item['ProductId']}_{idx}"):
                    del st.session_state['cart'][key]
                    # Remove item from Firestore
                    user_ref = db.collection("users").document(user_id)
                    cart_ref = user_ref.collection("cart").document(item['ProductId'])
                    cart_ref.delete()
                    st.success(f"**{item['ProductTitle']}** has been removed from your cart.")
        
        # Display the Confirm Purchase button only on the checkout page
        if st.button("Confirm Purchase", key="confirm_purchase"):
            missing_ratings = False
            for idx, key in enumerate(cart_keys):
                item = cart_items.get(key)
                if item:
                    rating_key = f"rate_{item['ProductId']}_{idx}"
                    rating = st.session_state.get(rating_key, None)
                    if rating is None or rating == 0:
                        st.warning(f"Please rate **{item['ProductTitle']}** before confirming the purchase.")
                        missing_ratings = True
                        break
            
            if not missing_ratings:
                for idx, key in enumerate(cart_keys):
                    item = cart_items.get(key)
                    rating_key = f"rate_{item['ProductId']}_{idx}"
                    rating = st.session_state.get(rating_key, 0)
                    if rating > 0:
                        save_feedback_to_firestore(db, item['ProductId'], item['ProductTitle'], user_id, rating)

                save_purchase_information(db, cart_items)
                st.success("Thank you for your purchase!")

                # Qualtrics survey link
                survey_url = "https://omaralanbari.com"
                st.markdown(f"Please proceed to our [Qualtrics survey]({survey_url}) to complete your experience. We appreciate your feedback!")

                st.session_state['cart'] = {}
                st.session_state['checkout'] = False  # Reset checkout state
    else:
        st.write("Your cart is empty.")
        st.session_state['checkout'] = False

    st.markdown("---")

    # Always display the "Go Back to Home Page" button
    if st.button("Go Back to Home Page", key="go_back_home_checkout"):
        st.session_state['checkout'] = False
        st.session_state['feedback_messages'] = []
        if 'clicked_product' in st.session_state:
            del st.session_state['clicked_product']
        st.rerun()
        logging.info(f"User {user_id} navigated back to home from checkout.")

    # Display feedback messages if they exist
    if st.session_state['feedback_messages']:
        combined_feedback = " | ".join(st.session_state['feedback_messages'])
        st.success(combined_feedback)

    # Stop execution to prevent home page from appearing
    st.stop()

elif "clicked_product" in st.session_state:
    product = st.session_state.clicked_product
    show_product_page(product)
    st.stop()
else:
    st.title("E-commerce Style Product Recommender")
    display_products(filtered_products_df)

# ============================
# Save Session Data at End of Script
# ============================
save_session_data()