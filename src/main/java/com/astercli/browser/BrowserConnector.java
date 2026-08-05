package com.astercli.browser;

public interface BrowserConnector {
    String status();

    String connectDefault();

    String disconnect();
}
