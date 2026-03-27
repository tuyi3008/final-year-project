# tests/test_integration_selenium.py
"""
Integration Test - Simulate real user operations using Selenium
"""

import pytest
import time
import os
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.common.exceptions import TimeoutException, NoAlertPresentException, ElementClickInterceptedException, UnexpectedAlertPresentException


class TestIntegration:
    """Selenium Integration Test Class"""
    
    @pytest.fixture
    def driver(self):
        """Launch Chrome browser"""
        chrome_options = Options()
        chrome_options.add_argument("--start-maximized")
        chrome_options.add_argument("--disable-infobars")
        chrome_options.add_argument("--disable-extensions")
        
        driver = webdriver.Chrome(
            service=Service(ChromeDriverManager().install()),
            options=chrome_options
        )
        driver.implicitly_wait(10)
        
        yield driver
        driver.quit()
    
    def wait_for_element(self, driver, by, value, timeout=10):
        """Wait for element to appear"""
        try:
            element = WebDriverWait(driver, timeout).until(
                EC.presence_of_element_located((by, value))
            )
            return element
        except TimeoutException:
            print(f"❌ Element not found: {by}={value}")
            return None
    
    def wait_for_clickable(self, driver, by, value, timeout=10):
        """Wait for element to be clickable"""
        try:
            element = WebDriverWait(driver, timeout).until(
                EC.element_to_be_clickable((by, value))
            )
            return element
        except TimeoutException:
            print(f"❌ Element not clickable: {by}={value}")
            return None
    
    def wait_for_modal(self, driver, timeout=10):
        """Wait for modal to appear"""
        try:
            modal = WebDriverWait(driver, timeout).until(
                EC.presence_of_element_located((By.CLASS_NAME, "modal.show"))
            )
            return modal
        except TimeoutException:
            print("❌ Modal not found")
            return None
    
    def handle_alert(self, driver, timeout=3):
        """Handle browser alert popup"""
        try:
            alert = WebDriverWait(driver, timeout).until(EC.alert_is_present())
            alert_text = alert.text
            print(f"⚠️ Alert detected: {alert_text}")
            alert.accept()
            print("✅ Alert closed")
            time.sleep(0.5)
            return True
        except:
            return False
    
    def test_complete_user_flow(self, driver):
        """Integration test: Complete user flow"""
        
        test_email = f"selenium_test_{int(time.time())}@test.com"
        test_username = f"testuser_{int(time.time())}"
        test_password = "test123456"
        
        print("\n" + "="*60)
        print("🚀 Starting Integration Test")
        print(f"📧 Test email: {test_email}")
        print(f"👤 Test username: {test_username}")
        print("="*60 + "\n")
        
        # Step 1: Open website
        print("📌 Step 1: Open website")
        driver.get("http://localhost:8000")
        time.sleep(2)
        print("✅ Website loaded\n")
        
        # Step 2: Open login modal
        print("📌 Step 2: Open login modal")
        signin_btn = self.wait_for_clickable(driver, By.CSS_SELECTOR, ".btn-login")
        signin_btn.click()
        print("✅ Clicked Sign In button")
        
        modal = self.wait_for_modal(driver)
        assert modal is not None, "Modal not found"
        print("✅ Modal opened\n")
        
        # Step 3: Switch to register tab
        print("📌 Step 3: Switch to register tab")
        register_tab = self.wait_for_clickable(driver, By.ID, "register-tab")
        register_tab.click()
        print("✅ Clicked register tab\n")
        
        # Step 4: Fill registration form
        print("📌 Step 4: Fill registration form")
        first_name = self.wait_for_element(driver, By.ID, "firstName")
        first_name.send_keys("Test")
        
        last_name = self.wait_for_element(driver, By.ID, "lastName")
        last_name.send_keys("User")
        
        email_input = self.wait_for_element(driver, By.ID, "registerEmail")
        email_input.send_keys(test_email)
        
        password_input = self.wait_for_element(driver, By.ID, "registerPassword")
        password_input.send_keys(test_password)
        
        confirm_input = self.wait_for_element(driver, By.ID, "confirmPassword")
        confirm_input.send_keys(test_password)
        
        terms_checkbox = self.wait_for_element(driver, By.ID, "termsAgree")
        if not terms_checkbox.is_selected():
            terms_checkbox.click()
        print("✅ Registration form filled\n")
        
        # Step 5: Submit registration and login
        print("📌 Step 5: Submit registration")
        submit_btn = self.wait_for_clickable(driver, By.CSS_SELECTOR, "#registerForm button[type='submit']")
        submit_btn.click()
        print("✅ Registration submitted")
        
        # Wait for registration to complete and modal to switch to login tab
        time.sleep(2)
        
        # Email should be pre-filled, just enter password
        print("📌 Enter password for login")
        login_password = self.wait_for_element(driver, By.ID, "loginPassword")
        login_password.send_keys(test_password)
        
        # Submit login immediately
        login_submit = self.wait_for_clickable(driver, By.CSS_SELECTOR, "#loginForm button[type='submit']")
        login_submit.click()
        print("✅ Login submitted")
        
        # Wait for modal to close
        WebDriverWait(driver, 5).until(
            EC.invisibility_of_element_located((By.CLASS_NAME, "modal.show"))
        )
        print("✅ Login successful, modal closed\n")
        
        # Step 6: Verify login status
        print("📌 Step 6: Verify login status")
        user_menu = driver.find_elements(By.CSS_SELECTOR, ".user-menu")
        if user_menu and len(user_menu) > 0 and user_menu[0].is_displayed():
            print("✅ User menu visible, login successful")
        
        user_icon = driver.find_elements(By.CSS_SELECTOR, ".user-icon")
        if user_icon:
            print("✅ User avatar visible\n")
        
        # Step 7: Navigate to gallery
        print("📌 Step 7: Navigate to gallery")
        gallery_link = self.wait_for_clickable(driver, By.XPATH, "//a[contains(@href, 'gallery')]")
        gallery_link.click()
        time.sleep(1)
        print(f"Current URL: {driver.current_url}\n")
        
        # Step 8: Navigate to profile
        print("📌 Step 8: Navigate to profile")
        driver.get("http://localhost:8000/static/profile.html")
        time.sleep(1)
        print(f"Current URL: {driver.current_url}\n")
        
        # Step 9: Logout
        print("📌 Step 9: Logout")
        logout_btn = self.wait_for_clickable(driver, By.CSS_SELECTOR, ".btn-logout", timeout=5)
        logout_btn.click()
        print("✅ Clicked logout button")
        time.sleep(1)
        
        self.handle_alert(driver)
        
        signin_btn = self.wait_for_element(driver, By.CSS_SELECTOR, ".btn-login", timeout=5)
        if signin_btn and signin_btn.is_displayed():
            print("✅ Logout successful\n")
        
        print("\n" + "="*60)
        print("🎉 Integration Test Complete!")
        print("="*60 + "\n")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])