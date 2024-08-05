from src.utils.real_time_processing import consume_real_time_data

if __name__ == "__main__":
    # Start real-time data processing in a separate thread or process
    import threading
    real_time_thread = threading.Thread(target=consume_real_time_data)
    real_time_thread.start()

    # Other parts of your recommendation engine can be initialized here
    print("Recommendation engine started")
