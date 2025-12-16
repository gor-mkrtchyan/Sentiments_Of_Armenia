CREATE TABLE geolocation (
    geolocation_id INT PRIMARY KEY,
    province VARCHAR(255),
    town_city VARCHAR(255)
);

CREATE TABLE locations (
    location_id INT PRIMARY KEY,
    geolocation_id INT REFERENCES geolocation(geolocation_id),
    location_type VARCHAR(20) CHECK (location_type IN ('Hotel','Restaurant')),
    location_name VARCHAR(255),
    location_rating DECIMAL(2,1),
    location_num_reviews INT,
    location_tag VARCHAR(255),
    location_price_range VARCHAR(50)
);

CREATE TABLE app_users (
    user_id INT PRIMARY KEY,
    user_name VARCHAR(255),
    user_from VARCHAR(255)
);

CREATE TABLE reviews (
    review_id INT PRIMARY KEY,
    location_id INT REFERENCES locations(location_id),
    user_id INT REFERENCES app_users(user_id),
    review_rating VARCHAR(10),
    review_subject VARCHAR(500),
    review_text TEXT,
    review_date DATE,
    review_type VARCHAR(255)
);

CREATE TABLE restaurant_review_details (
    review_id INT PRIMARY KEY REFERENCES reviews(review_id),
    value_rating VARCHAR(10),
    service_rating VARCHAR(10),
    food_rating VARCHAR(10),
    atmosphere_rating VARCHAR(10)
);

CREATE TABLE hotel_review_details (
    review_id INT PRIMARY KEY REFERENCES reviews(review_id),
    value_rating VARCHAR(10),
    rooms_rating VARCHAR(10),
    location_rating VARCHAR(10),
    cleanliness_rating VARCHAR(10),
    service_rating VARCHAR(10),
    sleep_quality_rating VARCHAR(10)
);