package main.org.backingdata.twitter.crawler.util;


import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.Properties;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Property manager
 * 
 * @author Francesco Ronzano
 *
 */
public class PropertyManager {
		
	private static final Logger logger = LoggerFactory.getLogger(PropertyManager.class);
	
	public static final String RESTtweetIDlistFilePath = "tweetID.fullPathTweetIDs";
	public static final String RESTtweetIDfullPathOfOutputDir = "tweetID.fullOutputDirPath";
	public static final String RESTtweetIDoutputFormat = "tweetID.outputFormat";
	
	public static final String RESTtweetKeywordListPath = "tweetKeyword.fullPathKeywordList";
	public static final String RESTtweetKeywordFullPathOfOutputDir = "tweetKeyword.fullOutputDirPath";
	public static final String RESTtweetKeywordOutputFormat = "tweetKeyword.outputFormat";
	public static final String RESTtweetKeywordLimitByLanguage = "tweetKeyword.languageFilter";
	
	public static final String RESTtweetTimelineListPath = "tweetTimeline.fullPathAccountIDs";
	public static final String RESTtweetTimelineFullPathOfOutputDir = "tweetTimeline.fullOutputDirPath";
	public static final String RESTtweetTimelineOutputFormat = "tweetTimeline.outputFormat";
	
	public static final String STREAMkeywordListPath = "tweetSTREAMkeyword.fullPathKeywordList";
	public static final String STREAMkeywordUserListPath = "tweetSTREAMkeyword.fullPathUserList";
	public static final String STREAMkeywordFullPathOfOutputDir = "tweetSTREAMkeyword.fullOutputDirPath";
	public static final String STREAMkeywordOutputFormat = "tweetSTREAMkeyword.outputFormat";
	public static final String STREAMkeywordLimitByLanguage = "tweetSTREAMkeyword.languageFilter";
	public static final String STREAMkeywordLimitByOneTweetPerXsec = "tweetSTREAMkeyword.limitByOneTweetPerXsec";
	public static final String STREAMkeywordFlushToFileEveryXtweetsCrawled = "tweetSTREAMkeyword.flushToFileEveryXtweetsCrawled";
	public static final String STREAMkeywordChangeStorageFileEveryXtweetsCrawled = "tweetSTREAMkeyword.changeStorageFileEveryXtweetsCrawled";
	
	public static final String STREAMbboxListPath = "tweetSTREAMbbox.fullPathBoundingBoxes";
	public static final String STREAMbboxFullPathOfOutputDir = "tweetSTREAMbbox.fullOutputDirPath";
	public static final String STREAMbboxOutputFormat = "tweetSTREAMbbox.outputFormat";
	public static final String STREAMbboxLimitByLanguage = "tweetSTREAMbbox.languageFilter";
	public static final String STREAMbboxLimitByOneTweetPerXsec = "tweetSTREAMbbox.limitByOneTweetPerXsec";
	public static final String STREAMbboxFlushToFileEveryXtweetsCrawled = "tweetSTREAMbbox.flushToFileEveryXtweetsCrawled";
	public static final String STREAMbboxChangeStorageFileEveryXtweetsCrawled = "tweetSTREAMbbox.changeStorageFileEveryXtweetsCrawled";
	
	private String propertyPath;
	private Properties holder = null; 
	
	
	public static final String defaultPropertyFilePath = "/local/path/to/crawler.properties";
	
	/**
	 * Load the property file.
	 * The path of the Text Digester property file is specified as a local absolute 
	 * (without trailing slash, for instance /home/mydir/TextDigesterConfig.properties)
	 * 
	 * @return
	 * @throws TextDigesterException 
	 * @throws InternalProcessingException
	 */
	public boolean loadProperties() throws Exception {
		
		FileInputStream input;
		try {
			input = new FileInputStream(propertyPath);
		} catch (FileNotFoundException e) {
			throw new Exception("PROPERTY FILE INITIALIZATION ERROR: property file '" + propertyPath + "' cannot be found");
		}
		
		try {
			holder = new Properties();
			holder.load(input);
		} catch (IOException e) {
			throw new Exception("PROPERTY FILE INITIALIZATION ERROR: property file '" + propertyPath + "' cannot be read (" +  e.getMessage() + ")");
		}
		
		return true;
	}
	
	/**
	 * Set the Text Digester property file path
	 * 
	 * @param filePath
	 * @return
	 */
	public boolean setPropertyFilePath(String filePath) {
		if(filePath != null) {
			File propFile = new File(filePath);
			if(propFile.exists() && propFile.isFile()) {
				propertyPath = filePath;
				return true;
			}
		}
		return false;
	}
	
	/**
	 * Retrieve a property from the Text Digester property file.
	 * The path of the Text Digester property file is specified as a local absolute 
	 * (without trailing slash, for instance /home/mydir/TextDigesterConfig.properties)
	 * 
	 * @param propertyName
	 * @return
	 * @throws Exception 
	 * @throws InternalProcessingException
	 */
	public String getProperty(String propertyName) throws Exception {
		if(propertyName != null && !propertyName.equals("")) {
			if(holder == null) {
				try {
					loadProperties();
				} catch (Exception e) {
					throw new Exception("Property file not correctly loaded");
				}
			}
			
			return holder.getProperty(propertyName);
		}
		else {
			return null;
		}
	}
	
	
}
