package main.org.backingdata.twitter.crawler.util;

import java.util.ArrayList;
import java.util.List;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class PropertyUtil {

	private static Logger logger = LoggerFactory.getLogger(PropertyUtil.class.getName());

	public static List<CredentialObject> loadCredentialObjects(PropertyManager propManager) {
		List<CredentialObject> credentialObjects = new ArrayList<CredentialObject>();

		try {
			if(propManager != null && propManager.loadProperties()) {

				for(int i = 0; i < 150; i++) {
					String keyPropStr = "consumerKey_" + i;
					String keyProp = propManager.getProperty(keyPropStr);
					if(keyProp != null && !keyProp.trim().equals("")) {
						
						System.out.println("\nReading credentials " + i + "...");
						
						String secrPropStr = "consumerSecret_" + i;
						String tokPropStr = "token_" + i;
						String tokSecrPropStr = "tokenSecret_" + i;
						
						String secrProp = propManager.getProperty(secrPropStr);
						String tokProp = propManager.getProperty(tokPropStr);
						String tokSecrProp = propManager.getProperty(tokSecrPropStr);
						
						System.out.println("     consumerKey_" + i + ": " + ((keyProp != null) ? keyProp.trim() : "NULL"));
						System.out.println("     consumerSecret_" + i + ": " + ((secrProp != null) ? secrProp.trim() : "NULL"));
						System.out.println("     token_" + i + ": " + ((tokProp != null) ? tokProp.trim() : "NULL"));
						System.out.println("     tokenSecret_" + i + ": " + ((tokSecrProp != null) ? tokSecrProp.trim() : "NULL"));
						
						if(secrProp != null && !secrProp.trim().equals("") && tokProp != null && !tokProp.trim().equals("") && tokSecrProp != null && !tokSecrProp.trim().equals("")) {
							CredentialObject newCredentials = new CredentialObject();
							newCredentials.setConsumerKey(keyProp.trim());
							newCredentials.setConsumerSecret(secrProp.trim());
							newCredentials.setToken(tokProp.trim());
							newCredentials.setTokenSecret(tokSecrProp.trim());
							
							// Check credential object
							String returnCheck = CredentialObject.validateCredentials(newCredentials);
							if(returnCheck == null) {
								newCredentials.setValid(true);
								newCredentials.setIsValidExplanation(null);
							}
							else {
								newCredentials.setValid(false);
								newCredentials.setIsValidExplanation(returnCheck);
							}
							
							credentialObjects.add(newCredentials);
							
						}
						else {
							System.out.println("     ERROR: invalid set of credentials");
						}
					}
				}
			}
		}
		catch (Exception e) {
			logger.error("Exception while loading properties: " + ((e.getMessage() != null) ? e.getMessage() : "NULL"));
		}


		return credentialObjects;
	}


}
