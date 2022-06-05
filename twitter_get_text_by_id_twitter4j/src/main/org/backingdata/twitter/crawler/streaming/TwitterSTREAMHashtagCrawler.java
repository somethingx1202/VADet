package main.org.backingdata.twitter.crawler.streaming;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.io.UnsupportedEncodingException;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import main.org.backingdata.twitter.crawler.util.CredentialObject;
import main.org.backingdata.twitter.crawler.util.PropertyUtil;
import main.org.backingdata.twitter.crawler.util.PropertyManager;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.twitter.hbc.ClientBuilder;
import com.twitter.hbc.core.Constants;
import com.twitter.hbc.core.endpoint.StatusesFilterEndpoint;
import com.twitter.hbc.core.processor.StringDelimitedProcessor;
import com.twitter.hbc.httpclient.BasicClient;
import com.twitter.hbc.httpclient.auth.Authentication;
import com.twitter.hbc.httpclient.auth.OAuth1;
import com.twitter.hbc.twitter4j.handler.StatusStreamHandler;
import com.twitter.hbc.twitter4j.message.DisconnectMessage;
import com.twitter.hbc.twitter4j.message.StallWarningMessage;

import twitter4j.StallWarning;
import twitter4j.Status;
import twitter4j.StatusDeletionNotice;
import twitter4j.StatusListener;
import twitter4j.TwitterException;
import twitter4j.json.DataObjectFactory;

/**
 * STREAMING Cralwer of Twitter - retrieves all tweets matching some specific keywords and/or in some specific language.<br/>
 * 
 * Have a look at the main code to identify how to define the set of keywords / languages and Twitter credentials.<br/><br/>
 * 
 * For each keywords, for every 20,000 tweet retrieved, a .txt file is created containing one JSON Tweet 
 * per line (https://dev.twitter.com/overview/api/tweets). The name of such file has the following format:<br/>
 * 
 * *SHARED_STRING*_*SEQUENCE_ID*_*KEYWORD*_from_*CRAWLING_START_DATE*.txt<br/><br/>
 * Log files with crawling errors and messages are also created.<br/>
 * 
 * 
 * @author Francesco Ronzano
 *
 */
public class TwitterSTREAMHashtagCrawler {

	private static Logger logger = LoggerFactory.getLogger(TwitterSTREAMHashtagCrawler.class.getName());

	private String fileSharedName = "twitter_stream_v1";

	// Authentication
	public List<String> consumerKey = new ArrayList<String>();
	public List<String> consumerSecret = new ArrayList<String>();
	public List<String> token = new ArrayList<String>();
	public List<String> tokenSecret = new ArrayList<String>();

	// Terms and language
	public List<String> trackTerms = new ArrayList<String>();
	public List<String> langList = new ArrayList<String>();
	public Map<String, Long> userMap = new HashMap<String, Long>();

	// Full local path of a local text file containing a list of tweet terms (one per line)
	private String fullPathOfTweetKeywordFile = "";

	// Full local path of a local text file containing a list of tweet terms (one per line)
	private String fullPathOfTweetTimelineFile = "";

	// Output directory
	private String outputDirPath = "";

	// Output format
	private String outpuTweetFormat = "";


	// Blocking queue for tweets to process
	private BlockingQueue<String> queue = new LinkedBlockingQueue<String>(10000);


	// Storage parameters
	private static File storageDir = null;
	private Map<String, PrintWriter> storageFileMap = new HashMap<String, PrintWriter>();
	private Map<String, Integer> storageFileCount = new HashMap<String, Integer>();
	private Map<String, Integer> storageFileId = new HashMap<String, Integer>();
	private Map<String, Long> storageFileLastTimestamp = new HashMap<String, Long>();

	private PrintWriter logFile = null;
	private Integer logFileId = 0;
	private Integer totalTweetStoredCount = 1;

	private Integer flushFileNumTweets = 100;
	private Integer flushLogFileNumMessages = 100;
	private Integer changeFileNumTweets = 20000;
	private Integer changeLogFileNumTweets = 80000;
	private Long storeMaxOneTweetEveryXseconds = -5l;

	// Date formatter
	private SimpleDateFormat sdf = new SimpleDateFormat("dd_M_yyyy__hh_mm_ss");

	// Appo vars
	private Map<String, List<String>> storageFileTweetList = new HashMap<String, List<String>>(); // Pointer to String list to store tweets of each location
	private List<String> logFileList = new ArrayList<String>();

	private long pastFreeMem = 0l;
	public void printMemoryStatus() {
		int MegaBytes = 10241024;
		long totalMemory = Runtime.getRuntime().totalMemory() / MegaBytes;
		long freeMemory = Runtime.getRuntime().freeMemory() / MegaBytes;
		long maxMemory = Runtime.getRuntime().maxMemory() / MegaBytes;

		System.out.println("Total heap size: " + totalMemory + " Mb of which: "
				+ " 1) free mem: " + freeMemory + " Mb (before: " + pastFreeMem + "Mb)"
				+ " 2) Max mem: " + maxMemory + " Mb");
		pastFreeMem = freeMemory;
	}

	public void checkLogAndSotrageFiles() {

		// Init maps
		if(storageFileMap.size() == 0) {
			for(String entry : trackTerms) {
				storageFileMap.put(entry, null);
			}
			for(Entry<String, Long> entry : userMap.entrySet()) {
				storageFileMap.put("ACCOUNT_" + entry.getKey().toLowerCase(), null);
			}
		}
		if(storageFileTweetList.size() == 0) {
			for(String entry : trackTerms) {
				storageFileTweetList.put(entry, new ArrayList<String>());
			}
			for(Entry<String, Long> entry : userMap.entrySet()) {
				storageFileTweetList.put("ACCOUNT_" + entry.getKey().toLowerCase(), new ArrayList<String>());
			}
		}
		if(storageFileCount.size() == 0) {
			for(String entry : trackTerms) {
				storageFileCount.put(entry, 1);
			}
			for(Entry<String, Long> entry : userMap.entrySet()) {
				storageFileCount.put("ACCOUNT_" + entry.getKey().toLowerCase(), 1);
			}
		}
		if(storageFileLastTimestamp.size() == 0) {
			for(String entry : trackTerms) {
				storageFileLastTimestamp.put(entry, 0l);
			}
			for(Entry<String, Long> entry : userMap.entrySet()) {
				storageFileLastTimestamp.put("ACCOUNT_" + entry.getKey().toLowerCase(), 0l);
			}
		}
		if(storageFileId.size() == 0) {
			for(String entry : trackTerms) {
				storageFileId.put(entry, 0);
			}
			for(Entry<String, Long> entry : userMap.entrySet()) {
				storageFileId.put("ACCOUNT_" + entry.getKey().toLowerCase(), 0);
			}
		}

		// If files are opened and the changeFileNumTweets is reached, these files are closed
		for(String entry : trackTerms) {
			String termString = entry;
			if( storageFileMap.containsKey(termString) && storageFileCount.containsKey(termString) && storageFileId.containsKey(termString) && storageFileMap.get(termString) != null) {

				boolean flushTweet = ((storageFileCount.get(termString) % this.flushFileNumTweets == 0) || (storageFileCount.get(termString) % this.changeFileNumTweets == 0)) ? true : false;
				boolean changeFile = (storageFileCount.get(termString) % this.changeFileNumTweets == 0) ? true : false;


				if(flushTweet && storageFileTweetList.get(termString).size() > 0) {
					// Store in tweet JSON file all messages in logFileList
					for(String storageFileMessage : storageFileTweetList.get(termString)) {
						if(storageFileMessage != null && storageFileMessage.trim().length() > 0) {
							storageFileMap.get(termString).write((storageFileMessage.endsWith("\n")) ? storageFileMessage : storageFileMessage + "\n");
						}
					}

					storageFileMap.get(termString).flush();
					System.out.println("Stored (flush) " + storageFileTweetList.get(termString).size() + " tweets to file of term  " + termString + " with id " + (storageFileId.get(termString) - 1));

					storageFileTweetList.put(termString, new ArrayList<String>());
				}

				if(changeFile) {
					storageFileMap.get(termString).flush();
					storageFileMap.get(termString).close();
					storageFileMap.put(termString, null);

					System.out.println("Closed tweets file of term  " + termString + " with id " + (storageFileId.get(termString) - 1));

					// Increase the number of storage count to not generate other files if the tweet is not stored
					Integer storageCountAppo = storageFileCount.get(termString);
					storageFileCount.put(termString, ++storageCountAppo);
				}

			}			
		}

		for(Entry<String, Long> entry : userMap.entrySet()) {
			String userString = entry.getKey().toLowerCase();
			if( storageFileMap.containsKey("ACCOUNT_" + userString) && storageFileCount.containsKey("ACCOUNT_" + userString) && storageFileId.containsKey("ACCOUNT_" + userString) ) {

				boolean flushTweet = ((storageFileCount.get("ACCOUNT_" + userString) % this.flushFileNumTweets == 0) || (storageFileCount.get("ACCOUNT_" + userString) % this.changeFileNumTweets == 0)) ? true : false;
				boolean changeFile = (storageFileCount.get("ACCOUNT_" + userString) % this.changeFileNumTweets == 0) ? true : false;


				if(flushTweet && storageFileTweetList.get("ACCOUNT_" + userString).size() > 0) {
					// Store in tweet JSON file all messages in logFileList
					for(String storageFileMessage : storageFileTweetList.get("ACCOUNT_" + userString)) {
						if(storageFileMessage != null && storageFileMessage.trim().length() > 0) {
							storageFileMap.get("ACCOUNT_" + userString).write((storageFileMessage.endsWith("\n")) ? storageFileMessage : storageFileMessage + "\n");
						}
					}

					storageFileMap.get("ACCOUNT_" + userString).flush();
					System.out.println("Stored (flush) " + storageFileTweetList.get("ACCOUNT_" + userString).size() + " tweets to file of account  " + userString + " with id " + (storageFileId.get("ACCOUNT_" + userString) - 1));

					storageFileTweetList.put("ACCOUNT_" + userString, new ArrayList<String>());
				}

				if(changeFile) {
					storageFileMap.get("ACCOUNT_" + userString).flush();
					storageFileMap.get("ACCOUNT_" + userString).close();
					storageFileMap.put("ACCOUNT_" + userString, null);

					System.out.println("Closed tweets file of account  " + userString + " with id " + (storageFileId.get("ACCOUNT_" + userString) - 1));

					// Increase the number of storage count to not generate other files if the tweet is not stored
					Integer storageCountAppo = storageFileCount.get("ACCOUNT_" + userString);
					storageFileCount.put("ACCOUNT_" + userString, ++storageCountAppo);
				}

			}
		}

		if(logFile != null && ( (totalTweetStoredCount % this.changeLogFileNumTweets) == 0) ) {
			boolean flushLog = ((totalTweetStoredCount % this.flushLogFileNumMessages == 0) || (totalTweetStoredCount % this.changeLogFileNumTweets == 0)) ? true : false;
			boolean changeLogFile = (totalTweetStoredCount % this.changeLogFileNumTweets == 0) ? true : false;


			if(flushLog) {
				// Store in log file all messages in logFileList
				for(String logFileMessage : logFileList) {
					if(logFileMessage != null && logFileMessage.trim().length() > 0) {
						logFile.write(logFileMessage);
					}
				}

				logFile.flush();
				System.out.println("Stored (flush) " + logFileList.size() + " log message to log file with id " + (logFileId - 1));

				logFileList = new ArrayList<String>();
			}

			if(changeLogFile) {
				logFile.flush();
				logFile.close();
				logFile = null;

				System.out.println("Closed log file with id " + (logFileId - 1));

				// Increase the number of storage count to not generate other log files if the tweet is not stored
				totalTweetStoredCount++;
			}

		}

		// Storage and log - open new files if null
		for(String entry : trackTerms) {
			String termString = entry;
			if( storageFileMap.containsKey(termString) && storageFileCount.containsKey(termString) && storageFileId.containsKey(termString) ) {

				if(storageFileMap.get(termString) == null) {
					String fileName = storageDir.getAbsolutePath() + File.separator + fileSharedName + "_TERM_" + termString + "_" + ( storageFileId.get(termString) ) + "_from_" + sdf.format(new Date()) + ".txt";
					Integer fileId = storageFileId.get(termString);
					storageFileId.put(termString, fileId + 1);
					storageFileLastTimestamp.put(termString, 0l);
					try {
						storageFileMap.put(termString, new PrintWriter(fileName, "UTF-8"));
					} catch (FileNotFoundException e) {
						logger.info("CANNOT OPEN FILE: " + fileName + " - Exception: " + e.getMessage());
						e.printStackTrace();
					} catch (UnsupportedEncodingException e) {
						logger.info("CANNOT OPEN FILE: " + fileName + " - Exception: " + e.getMessage());
						e.printStackTrace();
					}
				}
			}			
		}

		for(Entry<String, Long> entry : userMap.entrySet()) {
			String accountString = "ACCOUNT_" + entry.getKey().toLowerCase();
			if( storageFileMap.containsKey(accountString) && storageFileCount.containsKey(accountString) && storageFileId.containsKey(accountString) ) {

				if(storageFileMap.get(accountString) == null) {
					String fileName = storageDir.getAbsolutePath() + File.separator + fileSharedName + "_" + accountString + "_" + ( storageFileId.get(accountString) ) + "_from_" + sdf.format(new Date()) + ".txt";
					Integer fileId = storageFileId.get(accountString);
					storageFileId.put(accountString, fileId + 1);
					storageFileLastTimestamp.put(accountString, 0l);
					try {
						storageFileMap.put(accountString, new PrintWriter(fileName, "UTF-8"));
					} catch (FileNotFoundException e) {
						logger.info("CANNOT OPEN FILE: " + fileName + " - Exception: " + e.getMessage());
						e.printStackTrace();
					} catch (UnsupportedEncodingException e) {
						logger.info("CANNOT OPEN FILE: " + fileName + " - Exception: " + e.getMessage());
						e.printStackTrace();
					}
				}
			}			
		}

		// Log file	
		if(this.logFile == null) {
			String fileName = storageDir.getAbsolutePath() + File.separator + "LOG_" + fileSharedName + "_" + (logFileId++) + "_from_" + sdf.format(new Date()) + ".txt";
			try {
				this.logFile = new PrintWriter(fileName, "UTF-8");
			} catch (FileNotFoundException e) {
				logger.info("CANNOT OPEN LOG FILE: " + fileName + " - Exception: " + e.getMessage());
				e.printStackTrace();
			} catch (UnsupportedEncodingException e) {
				logger.info("CANNOT OPEN LOG FILE: " + fileName + " - Exception: " + e.getMessage());
				e.printStackTrace();
			}
		}

	}

	// Listener for Tweet stream
	@SuppressWarnings("unused")
	private StatusListener listener = new StatusStreamHandler() {

		public void onDeletionNotice(StatusDeletionNotice arg0) {
			// TODO Auto-generated method stub

		}

		public void onScrubGeo(long arg0, long arg1) {
			// TODO Auto-generated method stub

		}

		public void onStallWarning(StallWarning arg0) {
			// TODO Auto-generated method stub

		}

		public void onStatus(Status arg0) {
			// COPY CONTENTS BELOW:
			/*
			 * while( !queue.isEmpty() ) {...
			 */
		}

		public void onTrackLimitationNotice(int arg0) {
			checkLogAndSotrageFiles();

			try {
				logFileList.add((new Date()).toString() + " - TRACK LIMITATION NOTICE: " + arg0 + "\n");
				logger.info((new Date()).toString() + " - TRACK LIMITATION NOTICE: " + arg0 + "\n");
			} catch (Exception e) {
				System.out.println("Exception LOG FILE");
				e.printStackTrace();
			}
		}

		public void onException(Exception arg0) {
			checkLogAndSotrageFiles();

			try {
				logFileList.add((new Date()).toString() + " - EXCEPTION: " + arg0.getMessage() + "\n");
				logger.info((new Date()).toString() + " - EXCEPTION: " + arg0.getMessage() + "\n");
			} catch (Exception e) {
				System.out.println("Exception LOG FILE");
				e.printStackTrace();
			}

		}

		public void onDisconnectMessage(DisconnectMessage message) {
			checkLogAndSotrageFiles();

			try {
				logFileList.add((new Date()).toString() + " - DISCONNECT: CODE: " + message.getDisconnectCode() + ", REASON: " + message.getDisconnectReason() + "\n");
				logger.info((new Date()).toString() + " - DISCONNECT: CODE: " + message.getDisconnectCode() + ", REASON: " + message.getDisconnectReason() + "\n");
			} catch (Exception e) {
				System.out.println("Exception LOG FILE");
				e.printStackTrace();
			}
		}

		public void onStallWarningMessage(StallWarningMessage warning) {
			checkLogAndSotrageFiles();

			try {
				logFileList.add((new Date()).toString() + " - STALL WARNING: CODE: " + warning.hashCode() + ", REASON: " + warning.getMessage() + ", PERCENT FULL: " + warning.getPercentFull() + "\n");
				logger.info((new Date()).toString() + " - STALL WARNING: CODE: " + warning.hashCode() + ", REASON: " + warning.getMessage() + ", PERCENT FULL: " + warning.getPercentFull() + "\n");
			} catch (Exception e) {
				System.out.println("Exception LOG FILE");
				e.printStackTrace();
			}
		}

		public void onUnknownMessageType(String msg) {
			checkLogAndSotrageFiles();

			try {
				logFileList.add((new Date()).toString() + " - UNKNOWN MESSAGE: " + msg + "\n");
				logger.info((new Date()).toString() + " - UNKNOWN MESSAGE: " + msg + "\n");
			} catch (Exception e) {
				System.out.println("Exception LOG FILE");
				e.printStackTrace();
			}
		}
	};

	@SuppressWarnings("static-access")
	public void startCrawling() {
		StatusesFilterEndpoint endpoint = new StatusesFilterEndpoint();

		endpoint.trackTerms(this.trackTerms);
		logger.info("CRAWLING: " + this.trackTerms.size() + " TERMS:");
		for(String term : this.trackTerms) {
			logger.info("   TERM: " + term);
		}

		if(this.langList != null && this.langList.size() > 0) {
			endpoint.languages(this.langList);
			logger.info("CRAWLING: " + this.langList.size() + " LANGUAGES:");
			for(String language : this.langList) {
				logger.info("   LANGUAGE: " + language);
			}
		}

		if(this.userMap != null && this.userMap.size() > 0) {
			List<Long> userList = new ArrayList<Long>();
			for(Entry<String, Long> entry : this.userMap.entrySet()) {
				userList.add(new Long(entry.getValue()));
			}

			endpoint.followings(userList);
			logger.info("CRAWLING: " + userList.size() + " USERS:");
			for(Long user : userList) {
				logger.info("   USER: " + user);
			}
		}

		Authentication auth = new OAuth1(this.consumerKey.get(0), this.consumerSecret.get(0), this.token.get(0), this.tokenSecret.get(0));

		// Create a new BasicClient. By default gzip is enabled.
		BasicClient client = new ClientBuilder()
				.hosts(Constants.STREAM_HOST)
				.endpoint(endpoint)
				.authentication(auth)
				.processor(new StringDelimitedProcessor(queue))
				.build();

		// Establish a connection
		client.connect();

		// Loop to store tweets
		while(true) {
			this.checkLogAndSotrageFiles();

			while( !queue.isEmpty() ) {
				try {
					String msg;
					msg = queue.take();

					if(msg != null) {

						Integer indexOfLimit = msg.indexOf("\"limit\":");
						if(indexOfLimit >= 0 && indexOfLimit < 10) {
							logFileList.add(sdf.format(new Date()) + " - LIMIT STATUS: " + msg);
							logger.info(sdf.format(new Date()) + " - LIMIT STATUS: " + msg);	
							continue;
						}

						Integer indexOfLocationDeletion= msg.indexOf("\"scrub_geo\":");
						if(indexOfLocationDeletion >= 0 && indexOfLocationDeletion < 10) {
							logFileList.add(sdf.format(new Date()) + " - LOCATION DELETION STATUS: " + msg);
							logger.info(sdf.format(new Date()) + " - LOCATION DELETION STATUS: " + msg);
						}

						Integer indexOfStatusDeletion= msg.indexOf("\"delete\":");
						if(indexOfStatusDeletion >= 0 && indexOfStatusDeletion < 10) {
							logFileList.add(sdf.format(new Date()) + " - STATUS DELETION: " + msg);
							logger.info(sdf.format(new Date()) + " - STATUS DELETION: " + msg);
						}

						Integer indexOfStatusWithheld = msg.indexOf("\"status_withheld\":");
						if(indexOfStatusWithheld >= 0 && indexOfStatusWithheld < 10) {
							logFileList.add(sdf.format(new Date()) + " - STATUS WITHHELD: " + msg);
							logger.info(sdf.format(new Date()) + " - STATUS WITHHELD: " + msg);	
						}

						Integer indexOfUserWithheld = msg.indexOf("\"user_withheld\":");
						if(indexOfUserWithheld >= 0 && indexOfUserWithheld < 10) {
							logFileList.add(sdf.format(new Date()) + " - USER WITHHELD: " + msg);
							logger.info(sdf.format(new Date()) + " - USER WITHHELD: " + msg);	
						}

						Integer indexOfDisconect = msg.indexOf("\"disconnect\":");
						if(indexOfDisconect >= 0 && indexOfDisconect < 10) {
							logFileList.add(sdf.format(new Date()) + " - DISCONNECT STATUS: " + msg);
							logger.info(sdf.format(new Date()) + " - DISCONNECT STATUS: " + msg);	
						}

						Integer indexOfWarning = msg.indexOf("\"warning\":");
						if(indexOfWarning >= 0 && indexOfWarning < 10) {
							logFileList.add(sdf.format(new Date()) + " - WARNING STATUS: " + msg);
							logger.info(sdf.format(new Date()) + " - WARNING STATUS: " + msg);	
						}

					}

					Status receivedStatus = DataObjectFactory.createStatus(msg);
					if(receivedStatus != null && receivedStatus.getCreatedAt() != null) {

						this.checkLogAndSotrageFiles();

						try {

							// Save to account file
							long userId = receivedStatus.getUser().getId();
							String userName = "";
							for(Entry<String, Long> entry_user : this.userMap.entrySet()) {
								if(entry_user.getValue() == userId) {
									userName = entry_user.getKey().toLowerCase();
								}
							}

							if(userName != null && !userName.equals("")) {
								for(Map.Entry<String, PrintWriter> entry_int : storageFileMap.entrySet()) {
									if(entry_int.getKey().equals("ACCOUNT_" + userName)) {

										// Management of storeMaxOneTweetEveryXseconds
										if(storeMaxOneTweetEveryXseconds != null && storeMaxOneTweetEveryXseconds > 0l) {
											Long lastTimestamp = storageFileLastTimestamp.get("ACCOUNT_" + userName);
											if(lastTimestamp != null && (System.currentTimeMillis() - lastTimestamp) < (storeMaxOneTweetEveryXseconds * 1000l)) {
												System.out.println("SKIPPED TWEET OF USER: " + userName + " - only  " + (System.currentTimeMillis() - lastTimestamp) + " ms (< " + storeMaxOneTweetEveryXseconds + "s)"
														+ "since last tweet received - queue free places: " + queue.remainingCapacity());
												continue;
											}
											else {
												storageFileLastTimestamp.put("ACCOUNT_" + userName, System.currentTimeMillis());
											}
										}

										// Store to list
										storageFileTweetList.get(entry_int.getKey()).add(msg);
										// Store to file
										// entry_int.getValue().write(msg);
										// entry_int.getValue().flush();

										totalTweetStoredCount++;

										if(totalTweetStoredCount % 10 == 0) {
											printMemoryStatus();
											System.gc();
											System.out.println("GARBAGE COLLECTOR CALLED: ");
											printMemoryStatus();
										}

										for(Map.Entry<String, Integer> entry_int_int : storageFileCount.entrySet()) {
											if(entry_int_int.getKey().equals("ACCOUNT_" + userName)) {
												Integer storageCount = entry_int_int.getValue();
												storageFileCount.put("ACCOUNT_" + userName, storageCount + 1);
												System.out.println("RECEIVED: " + userName + " tot: " + (storageCount + 1) + " - queue free places: " + queue.remainingCapacity());
											}
										}
									}
								}	
							}

							// Save to term file
							for(Map.Entry<String, PrintWriter> entry_int : storageFileMap.entrySet()) {
								// If it is a term
								if(entry_int != null && entry_int.getKey() != null && !entry_int.getKey().startsWith("ACCOUNT_")) {
									String tweetTextLowercased = receivedStatus.getText().toLowerCase();
									String termLowercased = entry_int.getKey().toLowerCase();

									// Check if tweet contains the term
									boolean containsTerm = false;
									Pattern pattern = Pattern.compile("(^|[\\s|\\W])" + Pattern.quote(termLowercased) + "([\\s|\\W]|$)", Pattern.CASE_INSENSITIVE);
									Matcher matcher = pattern.matcher(tweetTextLowercased);
									Integer numMatches = 0;
									while (matcher.find()) {
										numMatches++;
									}

									if(numMatches > 0) {
										containsTerm = true;
									}

									// Store tweet if it contains the term
									if(containsTerm == true) {

										// Management of storeMaxOneTweetEveryXseconds
										if(storeMaxOneTweetEveryXseconds != null && storeMaxOneTweetEveryXseconds > 0l) {
											Long lastTimestamp = storageFileLastTimestamp.get(entry_int.getKey());
											if(lastTimestamp != null && (System.currentTimeMillis() - lastTimestamp) < (storeMaxOneTweetEveryXseconds * 1000l)) {
												// System.out.println("SKIPPED TWEET WITH TERM: " + entry_int.getKey() + " - only  " + (System.currentTimeMillis() - lastTimestamp) + " ms (< " + storeMaxOneTweetEveryXseconds + "s)"
												// 		+ "since last tweet received - queue free places: " + queue.remainingCapacity());
												continue;
											}
											else {
												storageFileLastTimestamp.put(entry_int.getKey(), System.currentTimeMillis());
											}
										}

										// Store to list
										storageFileTweetList.get(entry_int.getKey()).add(msg);
										// Store to file
										// entry_int.getValue().write(msg);
										// entry_int.getValue().flush();

										totalTweetStoredCount++;

										if(totalTweetStoredCount % 100 == 0) {
											printMemoryStatus();
											System.gc();
											System.out.println("GARBAGE COLLECTOR CALLED: ");
											printMemoryStatus();
										}

										for(Map.Entry<String, Integer> entry_int_int : storageFileCount.entrySet()) {
											if(entry_int_int.getKey().equals(entry_int.getKey())) {
												Integer storageCount = entry_int_int.getValue();
												storageFileCount.put(entry_int.getKey(), storageCount + 1);
												System.out.println("RECEIVED: " + entry_int.getKey() + " tot: " + (storageCount + 1) + " - queue free places: " + queue.remainingCapacity());
											}
										}
									}
								}
							}

						} catch (Exception e) {
							System.out.println("Exception " + e.getMessage());
							e.printStackTrace();
							logFileList.add(sdf.format(new Date()) + " - ERROR CODE: " + msg);
						}
					}
					else {
						logFileList.add(sdf.format(new Date()) + " - ERROR CODE: " + msg);
						logger.info(sdf.format(new Date()) + " - ERROR CODE: " + msg);	
						logFile.flush();
					}
				} catch (TwitterException e) {
					logger.info("ERROR WHILE PARSING TWEET: " + e.getMessage());
				} catch (InterruptedException e1) {
					logger.info("INTERRUPTED THREAD EXCEPTION: " + e1.getMessage());
				}
			}

			try {	
				Thread.currentThread().sleep(5000);
			} catch (InterruptedException e) {
				System.out.println("ERROR WHILE SLEEP PROCESSING MESSAGE THREAD: " + e.getMessage());
			}

		}
	}

	/**
	 * One argument is required, the full local path where data should be stored
	 * 
	 * @param args
	 */
	public static void main(String[] args) {

		if(args == null || args.length == 0 || args[0] == null || args[0].trim().equals("")) {
			System.out.println("Please, specify the full local path to the crawler property file as first argument!");
			return;
		}

		File crawlerPropertyFile = new File(args[0].trim());
		if(crawlerPropertyFile == null || !crawlerPropertyFile.exists() || !crawlerPropertyFile.isFile()) {
			System.out.println("The path of the crawler ptoperty file (first argument) is wrongly specified > PATH: '" + ((args[0] != null) ? args[0].trim() : "NULL") + "'");
			return;
		}

		TwitterSTREAMHashtagCrawler crawler = new TwitterSTREAMHashtagCrawler();

		// Load information from property file
		PropertyManager propManager = new PropertyManager();
		propManager.setPropertyFilePath(args[0].trim());

		// Load credential objects
		System.out.println("Loading twitter API credentials from the property file at '" + args[0].trim() + "':");
		List<CredentialObject> credentialObjList = PropertyUtil.loadCredentialObjects(propManager);
		if(credentialObjList != null && credentialObjList.size() > 0) {
			for(CredentialObject credentialObj : credentialObjList) {
				if(credentialObj != null && credentialObj.isValid()) {
					crawler.consumerKey.add(credentialObj.getConsumerKey());
					crawler.consumerSecret.add(credentialObj.getConsumerSecret());
					crawler.token.add(credentialObj.getToken());
					crawler.tokenSecret.add(credentialObj.getTokenSecret());
				}
				else {
					System.out.println("      - ERROR > INVALID CREDENTIAL SET: " + ((credentialObj != null) ? credentialObj.toString() : "NULL OBJECT"));
				}
			}
		}

		// Load full path of keyword list file
		try {
			String keywordlistFilePath = propManager.getProperty(PropertyManager.STREAMkeywordListPath);
			File tweetIDfile = new File(keywordlistFilePath);
			if(tweetIDfile == null || !tweetIDfile.exists() || !tweetIDfile.isFile()) {
				System.out.println("ERROR: Tweet ID input file path (property '" + PropertyManager.STREAMkeywordListPath + "')"
						+ " wrongly specified > PATH: '" + ((keywordlistFilePath != null) ? keywordlistFilePath : "NULL") + "'");
				if(tweetIDfile != null && !tweetIDfile.exists()) {
					System.out.println("      The file does not exist!"); 
				}
				if(tweetIDfile != null && tweetIDfile.exists() && !tweetIDfile.isFile()) {
					System.out.println("      The path does not point to a valid file!"); 
				}
			}
			else {
				crawler.fullPathOfTweetKeywordFile = keywordlistFilePath;
			}
		} catch (Exception e) {
			System.out.println("ERROR: keyword list input file path (property '" + PropertyManager.STREAMkeywordListPath + "')"
					+ " wrongly specified - exception: " + ((e.getMessage() != null) ? e.getMessage() : "NULL"));
		}


		// Load full path of users file
		try {
			String userlistFilePath = propManager.getProperty(PropertyManager.STREAMkeywordUserListPath);
			File tweetIDfile = new File(userlistFilePath);
			if(tweetIDfile == null || !tweetIDfile.exists() || !tweetIDfile.isFile()) {
				System.out.println("ERROR: Tweet ID input file path (property '" + PropertyManager.STREAMkeywordUserListPath + "')"
						+ " wrongly specified > PATH: '" + ((userlistFilePath != null) ? userlistFilePath : "NULL") + "'");
				if(tweetIDfile != null && !tweetIDfile.exists()) {
					System.out.println("      The file does not exist!"); 
				}
				if(tweetIDfile != null && tweetIDfile.exists() && !tweetIDfile.isFile()) {
					System.out.println("      The path does not point to a valid file!"); 
				}
			}
			else {
				crawler.fullPathOfTweetTimelineFile = userlistFilePath;
			}
		} catch (Exception e) {
			System.out.println("ERROR: user list input file path (property '" + PropertyManager.STREAMkeywordUserListPath + "')"
					+ " wrongly specified - exception: " + ((e.getMessage() != null) ? e.getMessage() : "NULL"));
		}


		// Load full path of output directory
		try {
			String outputDirectoryFilePath = propManager.getProperty(PropertyManager.STREAMkeywordFullPathOfOutputDir);
			File outputDirFile = new File(outputDirectoryFilePath);
			if(outputDirFile == null || !outputDirFile.exists() || !outputDirFile.isDirectory()) {
				System.out.println("ERROR: output directory full path (property '" + PropertyManager.STREAMkeywordFullPathOfOutputDir + "')"
						+ " wrongly specified > PATH: '" + ((outputDirectoryFilePath != null) ? outputDirectoryFilePath : "NULL") + "'");
				if(outputDirFile != null && !outputDirFile.exists()) {
					System.out.println("      The directory does not exist!"); 
				}
				if(outputDirFile != null && outputDirFile.exists() && !outputDirFile.isDirectory()) {
					System.out.println("      The path does not point to a valid directory!"); 
				}
				return;
			}
			else {
				crawler.outputDirPath = outputDirectoryFilePath;
			}
		} catch (Exception e) {
			System.out.println("ERROR: output directory full path (property '" + PropertyManager.STREAMkeywordFullPathOfOutputDir + "')"
					+ " wrongly specified - exception: " + ((e.getMessage() != null) ? e.getMessage() : "NULL"));
			return;
		}

		// Output format
		try {
			String outputFormat = propManager.getProperty(PropertyManager.STREAMkeywordOutputFormat);

			if(outputFormat != null && outputFormat.trim().toLowerCase().equals("json")) {
				crawler.outpuTweetFormat = "json";
			}
			else if(outputFormat != null && outputFormat.trim().toLowerCase().equals("tab")) {
				crawler.outpuTweetFormat = "tab";
			}
			else {
				crawler.outpuTweetFormat = "json";
				System.out.println("Impossible to read the '" + PropertyManager.STREAMkeywordOutputFormat + "' property - set to: " + crawler.outpuTweetFormat);
			}

		} catch (Exception e) {
			System.out.println("ERROR: output format (property '" + PropertyManager.STREAMkeywordOutputFormat + "') - exception: " + ((e.getMessage() != null) ? e.getMessage() : "NULL"));
			return;
		}


		// Limit by one tweet per X seconds
		try {
			String limitRate = propManager.getProperty(PropertyManager.STREAMkeywordLimitByOneTweetPerXsec);

			if(limitRate != null && !limitRate.trim().equals("")) {
				try {
					crawler.storeMaxOneTweetEveryXseconds = Long.valueOf(limitRate);
				}
				catch(Exception e) {
					crawler.storeMaxOneTweetEveryXseconds = -1l;
					System.out.println("Impossible to read the '" + PropertyManager.STREAMkeywordLimitByOneTweetPerXsec + "' property - set to: " + crawler.storeMaxOneTweetEveryXseconds);
				}
			}
			else {
				crawler.storeMaxOneTweetEveryXseconds = -1l;
			}

		} catch (Exception e) {
			System.out.println("ERROR: output format (property '" + PropertyManager.STREAMkeywordLimitByOneTweetPerXsec + "') - exception: " + ((e.getMessage() != null) ? e.getMessage() : "NULL"));
			return;
		}



		// Language filter
		try {
			String langFilter = propManager.getProperty(PropertyManager.STREAMkeywordLimitByLanguage);

			if(langFilter != null) {
				String[] langArray = langFilter.split(",");

				for(String lang : langArray) {
					if(lang != null && !lang.equals("")) {
						crawler.langList.add(lang);
					}
				}
			}
			else {
				crawler.langList = new ArrayList<String>();
				System.out.println("Impossible to read the '" + PropertyManager.STREAMkeywordLimitByLanguage + "' property - Language filter not set");
			}

		} catch (Exception e) {
			System.out.println("ERROR: output format (property '" + PropertyManager.STREAMkeywordLimitByLanguage + "') - exception: " + ((e.getMessage() != null) ? e.getMessage() : "NULL"));
			return;
		}

		// Flush tweet to files every X tweet received
		try {
			String flushRate = propManager.getProperty(PropertyManager.STREAMkeywordFlushToFileEveryXtweetsCrawled);

			if(flushRate != null && !flushRate.trim().equals("")) {
				try {
					crawler.flushFileNumTweets = Integer.valueOf(flushRate);
					
					/*
					if(crawler.flushFileNumTweets < 20) {
						crawler.flushFileNumTweets = 20;
					}
					*/
				}
				catch(Exception e) {
					crawler.flushFileNumTweets = 100;
					System.out.println("Impossible to read the '" + PropertyManager.STREAMkeywordFlushToFileEveryXtweetsCrawled + "' property - set to: " + crawler.flushFileNumTweets);
				}
			}
			else {
				crawler.flushFileNumTweets = 100;
			}

		} catch (Exception e) {
			System.out.println("ERROR: output format (property '" + PropertyManager.STREAMkeywordFlushToFileEveryXtweetsCrawled + "') - exception: " + ((e.getMessage() != null) ? e.getMessage() : "NULL"));
			return;
		}

		// Change storage file every X tweet received
		try {
			String changeFileRate = propManager.getProperty(PropertyManager.STREAMkeywordChangeStorageFileEveryXtweetsCrawled);

			if(changeFileRate != null && !changeFileRate.trim().equals("")) {
				try {
					crawler.changeFileNumTweets = Integer.valueOf(changeFileRate);

					if(crawler.changeFileNumTweets < 200) {
						crawler.changeFileNumTweets = 200;
					}
				}
				catch(Exception e) {
					crawler.changeFileNumTweets = 20000;
					System.out.println("Impossible to read the '" + PropertyManager.STREAMkeywordChangeStorageFileEveryXtweetsCrawled + "' property - set to: " + crawler.changeFileNumTweets);
				}
			}
			else {
				crawler.changeFileNumTweets = 20000;
			}

		} catch (Exception e) {
			System.out.println("ERROR: output format (property '" + PropertyManager.STREAMkeywordChangeStorageFileEveryXtweetsCrawled + "') - exception: " + ((e.getMessage() != null) ? e.getMessage() : "NULL"));
			return;
		}

		if((crawler.fullPathOfTweetKeywordFile == null || crawler.fullPathOfTweetKeywordFile.equals("")) && 
				(crawler.fullPathOfTweetTimelineFile == null || crawler.fullPathOfTweetTimelineFile.equals(""))) {
			System.out.println("ERROR! USER AND KEYWORD LIST EMPTY! IMPOSSIBLE TO CRAWL DATA");
			return;
		}

		// Loading tweet keywords from file
		try {
			BufferedReader in = new BufferedReader(new InputStreamReader(new FileInputStream(new File(crawler.fullPathOfTweetKeywordFile)), "UTF-8"));

			String str;
			while ((str = in.readLine()) != null) {
				if(!str.trim().equals("")) {
					crawler.trackTerms.add(str.trim());
				}
			}

			in.close();
		}
		catch (Exception e) {
			System.out.println("Exception reading keywords from file: " +  e.getMessage() + " > PATH: '" + ((crawler.fullPathOfTweetKeywordFile != null) ? crawler.fullPathOfTweetKeywordFile : "NULL") + "'");
		}


		// Loading tweet keywords from file
		// Per line ACCOUNT NAME <TAB> ACCOUNT_ID_LONG
		try {
			BufferedReader in = new BufferedReader(new InputStreamReader(new FileInputStream(new File(crawler.fullPathOfTweetTimelineFile)), "UTF-8"));

			String str;
			while ((str = in.readLine()) != null) {
				if(!str.trim().equals("")) {
					try {
						String[] strDiv = str.trim().split("\t");

						crawler.userMap.put(strDiv[0], Long.valueOf(strDiv[1].trim()));
					}
					catch(Exception e) {
						e.printStackTrace();
						System.out.println("Impossible to parse the account line:' " + str + "' from file: '" + crawler.fullPathOfTweetTimelineFile + "'");
					}
				}
			}

			in.close();
		}
		catch (Exception e) {
			System.out.println("Exception reading tweet accounts from file: " +  e.getMessage() + " > PATH: '" + ((crawler.fullPathOfTweetTimelineFile != null) ? crawler.fullPathOfTweetTimelineFile : "NULL") + "'");
		}

		crawler.storageDir = new File(crawler.outputDirPath);

		// Printing arguments:
		System.out.println("\n***************************************************************************************");
		System.out.println("******************** LOADED PARAMETERS ************************************************");
		System.out.println("   > Property file loaded from path: '" + ((args[0].trim() != null) ? args[0].trim() : "NULL") + "'");
		System.out.println("        PROPERTIES:");
		System.out.println("           - NUMBER OF TWITTER API CREDENTIALS: " + ((crawler.consumerKey != null) ? crawler.consumerKey.size() : "ERROR"));
		System.out.println("           - LANGUAGE FILTER: " + ((crawler.langList != null) ? crawler.langList : "ERROR"));
		System.out.println("           - PATH OF LIST OF KEYWORD TO CRAWL: '" + ((crawler.fullPathOfTweetKeywordFile != null) ? crawler.fullPathOfTweetKeywordFile : "NULL") + "'");
		System.out.println("           - PATH OF LIST OF USER ACCOUNTS TO CRAWL: '" + ((crawler.fullPathOfTweetTimelineFile != null) ? crawler.fullPathOfTweetTimelineFile : "NULL") + "'");
		System.out.println("           - PATH OF CRAWLER OUTPUT FOLDER: '" + ((crawler.outputDirPath != null) ? crawler.outputDirPath : "NULL") + "'");
		System.out.println("           - OUTPUT FORMAT: '" + ((crawler.outputDirPath != null) ? crawler.outputDirPath : "NULL") + "'");
		System.out.println("           - STORE TWEETS TO FILE EVERY " + ((crawler.flushFileNumTweets != null) ? crawler.flushFileNumTweets : "NULL") + " TWEETS CRAWLED (MIN. ALLOWED VAL 20, DEFAULT VALUE 100)");
		System.out.println("           - SWITCH TO NEW TWEETS STORAGE FILE EVERY " + ((crawler.changeFileNumTweets != null) ? crawler.changeFileNumTweets : "NULL") + " TWEETS CRAWLED (MIN. ALLOWED VAL 200, DEFAULT VALUE 20000)");
		System.out.println("               (STORAGE FILES AND COUNTERS OF CRAWLED TWEETS ARE MANAGED SEPARATELY, ONE FOR EACH TERM OR USER ACCOUNT)");
		System.out.println("   -");
		System.out.println("   NUMBER OF TWEET KEYWORDS / LINES READ FROM THE LIST: " + ((crawler.trackTerms != null) ? crawler.trackTerms.size() : "READING ERROR"));
		System.out.println("   NUMBER OF TWEET USERS / LINES READ FROM THE LIST: " + ((crawler.userMap != null) ? crawler.userMap.size() : "READING ERROR"));
		System.out.println("   -");
		System.out.println("   LIMIT BY ONE TWEET PER X SECONDS SET TO: " + ((crawler.storeMaxOneTweetEveryXseconds != null) ? crawler.storeMaxOneTweetEveryXseconds : "READING ERROR"));
		System.out.println("***************************************************************************************\n");		

		if(crawler.trackTerms == null || crawler.trackTerms.size() == 0) {
			System.out.println("Empty list of Tweet keyword to crawl > EXIT");
			return;
		}

		if(crawler.consumerKey == null || crawler.consumerKey.size() == 0) {
			System.out.println("Empty list of valid Twitter API credentials > EXIT");
			return;
		}

		System.out.println("<><><><><><><><><><><><><><><><><><><>");
		System.out.println("List of keywords to crawl:");
		int keywordCounter = 1;
		for(String keyword : crawler.trackTerms) {
			System.out.println(keywordCounter++ + " keyword: " + keyword);
		}

		System.out.println("\nList of users to crawl:");
		int userCounter = 1;
		for(Entry<String, Long> userEntry : crawler.userMap.entrySet()) {
			System.out.println(userCounter++ + " Username: " + userEntry.getKey() + " > " + userEntry.getValue());
		}
		System.out.println("<><><><><><><><><><><><><><><><><><><>");

		try {
			Thread.sleep(4000);
		} catch (InterruptedException e) {
			/* Do nothing */
		}

		crawler.startCrawling();
	}

}
