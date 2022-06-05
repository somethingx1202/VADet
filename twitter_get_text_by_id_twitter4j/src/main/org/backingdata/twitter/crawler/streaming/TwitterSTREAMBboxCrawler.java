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

import main.org.backingdata.twitter.crawler.streaming.model.Bbox;
import main.org.backingdata.twitter.crawler.util.CredentialObject;
import main.org.backingdata.twitter.crawler.util.PropertyUtil;
import main.org.backingdata.twitter.crawler.util.PropertyManager;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.twitter.hbc.ClientBuilder;
import com.twitter.hbc.core.Constants;
import com.twitter.hbc.core.endpoint.Location;
import com.twitter.hbc.core.endpoint.Location.Coordinate;
import com.twitter.hbc.core.endpoint.StatusesFilterEndpoint;
import com.twitter.hbc.core.processor.StringDelimitedProcessor;
import com.twitter.hbc.httpclient.BasicClient;
import com.twitter.hbc.httpclient.auth.Authentication;
import com.twitter.hbc.httpclient.auth.OAuth1;
import com.twitter.hbc.twitter4j.handler.StatusStreamHandler;
import com.twitter.hbc.twitter4j.message.DisconnectMessage;
import com.twitter.hbc.twitter4j.message.StallWarningMessage;

import twitter4j.GeoLocation;
import twitter4j.StallWarning;
import twitter4j.Status;
import twitter4j.StatusDeletionNotice;
import twitter4j.StatusListener;
import twitter4j.TwitterException;
import twitter4j.json.DataObjectFactory;


/**
 * STREAMING Cralwer of Twitter - retrieves all tweets with geocoordinates / issued from a venue intersecting a set of bounding boxes.<br/>
 * 
 * Only tweets issued by venues with an area < 100kmq are considered.<br/>
 * 
 * Have a look at the main code to identify how to define the set of bounding boxes to crawl and the respective geo-coordinates as well as the Twitter credentials.<br/><br/>
 * 
 * For each bounding box, for every 20,000 tweet retrieved, a .txt file is created containing one JSON Tweet 
 * per line (https://dev.twitter.com/overview/api/tweets). The name of such file has the following format 
 * (tweets are stored in memory and then when 20,000 tweets are gathered all the bath of JSON is stored to the file):<br/>
 * 
 * *SHARED_STRING*_*SEQUENCE_ID*_*BOUNDING_BOX_NAME*_from_*CRAWLING_START_DATE*.txt<br/><br/>
 * Log files with crawling errors and messages are also created.<br/>
 * 
 * 
 * @author Francesco Ronzano
 *
 */
public class TwitterSTREAMBboxCrawler {

	private static Logger logger = LoggerFactory.getLogger(TwitterSTREAMBboxCrawler.class.getName());

	private String fileSharedName = "twitter_bbox_v1";
	private String fullPathOfBoundingBoxesFile = "";

	// Authentication
	public List<String> consumerKey = new ArrayList<String>();
	public List<String> consumerSecret = new ArrayList<String>();
	public List<String> token = new ArrayList<String>();
	public List<String> tokenSecret = new ArrayList<String>();

	// List of BBOX to crawl
	public Map<String, Bbox> trackBbox = new HashMap<String, Bbox>();

	// Blocking queue for tweets to process
	private BlockingQueue<String> queue = new LinkedBlockingQueue<String>(10000);

	// Output directory
	private String outputDirPath = "";

	// Output format
	private String outpuTweetFormat = "";

	// Language list
	public List<String> langList = new ArrayList<String>();

	// Storage parameters
	private static File outputDir = null;
	private Map<String, PrintWriter> storageFileMap = new HashMap<String, PrintWriter>(); // Pointer to file to store tweets of each location
	private Map<String, Integer> storageFileCount = new HashMap<String, Integer>(); // Counter of tweets stored for each location
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

	public synchronized void checkLogAndSotrageFiles() {

		// Init maps
		if(storageFileMap.size() == 0) {
			for(Map.Entry<String, Bbox> entry : trackBbox.entrySet()) {
				String key = entry.getKey();
				storageFileMap.put(key, null);
			}
		}
		if(storageFileTweetList.size() == 0) {
			for(Map.Entry<String, Bbox> entry : trackBbox.entrySet()) {
				String key = entry.getKey();
				storageFileTweetList.put(key, new ArrayList<String>());
			}
		}
		if(storageFileCount.size() == 0) {
			for(Map.Entry<String, Bbox> entry : trackBbox.entrySet()) {
				String key = entry.getKey();
				storageFileCount.put(key, 1);
			}
		}
		if(storageFileLastTimestamp.size() == 0) {
			for(Map.Entry<String, Bbox> entry : trackBbox.entrySet()) {
				String key = entry.getKey();
				storageFileLastTimestamp.put(key, 0l);
			}
		}
		if(storageFileId.size() == 0) {
			for(Map.Entry<String, Bbox> entry : trackBbox.entrySet()) {
				String key = entry.getKey();
				storageFileId.put(key, 0);
			}
		}

		// Check if to flush tweets to file or change storage file
		for(Map.Entry<String, Bbox> entry : trackBbox.entrySet()) {
			String locationString = entry.getKey();
			if( storageFileMap.containsKey(locationString) && storageFileCount.containsKey(locationString) && storageFileId.containsKey(locationString) && storageFileMap.get(locationString) != null) {

				boolean flushTweet = ((storageFileCount.get(locationString) % this.flushFileNumTweets == 0) || (storageFileCount.get(locationString) % this.changeFileNumTweets == 0)) ? true : false;
				boolean changeFile = (storageFileCount.get(locationString) % this.changeFileNumTweets == 0) ? true : false;

				if(flushTweet && storageFileTweetList.get(locationString).size() > 0) {
					// Store in tweet JSON file all messages in logFileList
					for(String storageFileMessage : storageFileTweetList.get(locationString)) {
						if(storageFileMessage != null && storageFileMessage.trim().length() > 0) {
							storageFileMap.get(locationString).write((storageFileMessage.endsWith("\n")) ? storageFileMessage : storageFileMessage + "\n");
						}
					}

					storageFileMap.get(locationString).flush();
					System.out.println("Stored (flush) " + storageFileTweetList.get(locationString).size() + " tweets to file of location  " + locationString + " with id " + (storageFileId.get(locationString) - 1));

					storageFileTweetList.put(locationString, new ArrayList<String>());
				}

				if(changeFile) {
					storageFileMap.get(locationString).flush();
					storageFileMap.get(locationString).close();
					storageFileMap.put(locationString, null);

					System.out.println("Closed tweets file of location  " + locationString + " with id " + (storageFileId.get(locationString) - 1));

					// Increase the number of storage count to not generate other files if the tweet is not stored
					Integer storageCountAppo = storageFileCount.get(locationString);
					storageFileCount.put(locationString, ++storageCountAppo);
				}
			}			
		}



		if(logFile != null) {

			boolean flushLog = ((totalTweetStoredCount % this.flushLogFileNumMessages == 0) || (totalTweetStoredCount % this.changeLogFileNumTweets == 0)) ? true : false;
			boolean changeLogFile = (totalTweetStoredCount % this.changeLogFileNumTweets == 0) ? true : false;


			if(flushLog && logFileList.size() > 0) {
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
		for(Map.Entry<String, Bbox> entry : trackBbox.entrySet()) {
			String locationString = entry.getKey();
			if( storageFileMap.containsKey(locationString) && storageFileCount.containsKey(locationString) && storageFileId.containsKey(locationString) ) {

				if(storageFileMap.get(locationString) == null) {
					String fileName = outputDir.getAbsolutePath() + File.separator + fileSharedName + "_" + ( storageFileId.get(locationString) ) + "_" + locationString + "_from_" + sdf.format(new Date()) + ".txt";
					Integer fileId = storageFileId.get(locationString);
					storageFileId.put(locationString, fileId + 1);
					try {
						storageFileMap.put(locationString, new PrintWriter(fileName, "UTF-8"));
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

		if(logFile == null) {
			String fileName = outputDir.getAbsolutePath() + File.separator + "LOG_" + fileSharedName + "_" + ( logFileId++ ) + "_from_" + sdf.format(new Date()) + ".txt";
			try {
				logFile = new PrintWriter(fileName, "UTF-8");
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

	@SuppressWarnings("static-access")
	public void startCrawling() {
		StatusesFilterEndpoint endpoint = new StatusesFilterEndpoint();

		List<Location> locList = new ArrayList<Location>();
		for(Map.Entry<String, Bbox> entry : this.trackBbox.entrySet()) {
			Bbox bb = entry.getValue();
			if(bb != null && bb.getLngSW() != 0d && bb.getLatSW() != 0d && bb.getLngNE() != 0d && bb.getLatNE() != 0d) {
				Coordinate swCoord = new Coordinate(bb.getLngSW(), bb.getLatSW());
				Coordinate neCoord = new Coordinate(bb.getLngNE(), bb.getLatNE());
				Location loc = new Location(swCoord, neCoord);
				locList.add(loc);
			}
		}

		endpoint.locations(locList);

		if(this.langList != null && this.langList.size() > 0) {
			endpoint.languages(this.langList);
			logger.info("CRAWLING: " + this.langList.size() + " LANGUAGES:");
			for(String language : this.langList) {
				logger.info("   LANGUAGE: " + language);
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
					Status receivedStatus = DataObjectFactory.createStatus(msg);
					if(receivedStatus != null && receivedStatus.getCreatedAt() != null) {

						this.checkLogAndSotrageFiles();

						try {
							// Check if the tweets has coordinates
							if(receivedStatus != null && receivedStatus.getGeoLocation() != null) {
								double latitudeVal = receivedStatus.getGeoLocation().getLatitude();
								double longitudeVal = receivedStatus.getGeoLocation().getLongitude();

								if(-90d <= latitudeVal && latitudeVal <= 90d && -180d <= longitudeVal && longitudeVal <= 180d) {

									boolean inOneBbox = false;

									for(Map.Entry<String, Bbox> entry : trackBbox.entrySet()) {
										String locationString = entry.getKey();
										Bbox bb = entry.getValue();
										if(bb != null && bb.isInBbox(longitudeVal, latitudeVal)) {
											for(Map.Entry<String, PrintWriter> entry_int : storageFileMap.entrySet()) {
												if(entry_int.getKey().equals(locationString)) {

													for(Map.Entry<String, Integer> entry_int_int : storageFileCount.entrySet()) {
														if(entry_int_int.getKey().equals(locationString)) {

															boolean storeTweet = false;

															// Management of storeMaxOneTweetEveryXseconds
															if(storeMaxOneTweetEveryXseconds != null && storeMaxOneTweetEveryXseconds > 0l) {
																Long lastTimestamp = storageFileLastTimestamp.get(locationString);
																if(lastTimestamp != null && (System.currentTimeMillis() - lastTimestamp) < (storeMaxOneTweetEveryXseconds * 1000l)) {
																	System.out.println("SKIPPED TWEET FOR LOCATION: " + locationString + " - only  " + (System.currentTimeMillis() - lastTimestamp) + " ms (< " + storeMaxOneTweetEveryXseconds + "s)"
																			+ "since last tweet received - queue free places: " + queue.remainingCapacity());
																	storeTweet = false;
																}
																else {
																	storageFileLastTimestamp.put(locationString, System.currentTimeMillis());
																	storeTweet = true;
																}
															}
															else {
																// No storeMaxOneTweetEveryXseconds set, store tweet anyway
																storeTweet = true;
															}

															if(storeTweet) {
																// Store to list
																storageFileTweetList.get(entry_int.getKey()).add(msg);
																// Store to file
																// entry_int.getValue().write(msg);
																// entry_int.getValue().flush();

																inOneBbox = true;
																totalTweetStoredCount++;

																// Increment number of stored tweets for the bounding box
																Integer storageCount = entry_int_int.getValue();
																storageFileCount.put(locationString, storageCount + 1);
																System.out.println("RECEIVED (lat, lng) GEOLOCATED TWEET: " + locationString + " tot: " + (storageCount + 1) + " - queue free places: " + queue.remainingCapacity());
															}

														}
													}

													if(totalTweetStoredCount % 100 == 0) {
														printMemoryStatus();
														System.gc();
														System.out.println("GARBAGE COLLECTOR CALLED: ");
														printMemoryStatus();
													}

												}
											}							
										}
									}

									if(!inOneBbox) {
										logFileList.add("TWEET outside bboxes (lng: " + longitudeVal + ", lat: " + latitudeVal + ")\n");
										// System.out.println("WARN: TWEET outside bboxes (lng: " + longitudeVal + ", lat: " + latitudeVal + ") - queue free places: " + queue.remainingCapacity());
									}
								}
								else {
									logFileList.add("TWEET WRONG GEOCOORDINATES\n");
									// System.out.println("WARN: RECEIVED WRONG GEOCOORDINATES - queue free places: " + queue.remainingCapacity());
								}

							}
							else {
								// If there is a place intersecting
								if(receivedStatus != null && receivedStatus.getPlace() != null && receivedStatus.getPlace().getBoundingBoxCoordinates() != null) {
									GeoLocation[][] placeCoordinates = receivedStatus.getPlace().getBoundingBoxCoordinates();
									if(placeCoordinates != null && placeCoordinates.length > 0) {

										if(placeCoordinates.length >= 1) {
											GeoLocation[] plCoord = placeCoordinates[0];

											if(plCoord != null && plCoord.length == 4) {
												double minLat = Double.MAX_VALUE;
												double maxLat = Double.MIN_VALUE;
												double minLon= Double.MAX_VALUE;
												double maxLon = Double.MIN_VALUE;
												for(int j = 0; j < plCoord.length; j++) {
													GeoLocation plCoordInt = placeCoordinates[0][j];
													double latitude = plCoordInt.getLatitude();
													double longitude = plCoordInt.getLongitude();
													if(latitude < minLat) {
														minLat = latitude;
													}
													if(latitude > maxLat) {
														maxLat = latitude;
													}
													if(longitude < minLon) {
														minLon = longitude;
													}
													if(longitude > maxLon) {
														maxLon = longitude;
													}

													// System.out.println("       > POINT: (" + plCoordInt.getLatitude() + ", " + plCoordInt.getLongitude() + ")");
												}

												if(minLat != Double.MAX_VALUE && minLon != Double.MAX_VALUE && maxLat != Double.MIN_VALUE && maxLon != Double.MIN_VALUE) {
													double latitudeDiff = 0d;
													double longitudeDiff = 0d;

													if(maxLat >= 0d && minLat >= 0d) {
														latitudeDiff = maxLat - minLat;
													}
													else if(maxLat <= 0d && minLat <= 0d) {
														latitudeDiff = (-minLat) - (-maxLat);
													}
													else if(maxLat >= 0d && minLat <= 0d) {
														latitudeDiff = maxLat + (-minLat);
													}	


													if(maxLon >= 0d && minLon >= 0d) {
														longitudeDiff = maxLon - minLon;
													}
													else if(maxLon <= 0d && minLon <= 0d) {
														longitudeDiff = (-minLon) - (-maxLon);
													}
													else if(maxLon >= 0d && minLon <= 0d) {
														longitudeDiff = maxLon + (-minLon);
													}

													if(latitudeDiff > 0d && longitudeDiff > 0d) {
														latitudeDiff = (latitudeDiff / Bbox.constAreaKm);
														longitudeDiff = (longitudeDiff / Bbox.constAreaKm);
														double areaPlace = latitudeDiff * longitudeDiff;

														double areaInKmQuad = areaPlace;
														double maxAreaInKmQuad = 100d;

														if(areaInKmQuad <= 100d ) {

															// Check if one corner of the place BBOX is inside one of the BBOXes to crawl
															boolean inOneBbox = false;

															for(Map.Entry<String, Bbox> entry : trackBbox.entrySet()) {
																String locationString = entry.getKey();
																Bbox bb = entry.getValue();

																if(bb != null && 
																		(bb.isInBbox(minLon, minLat) || bb.isInBbox(minLon, maxLat) || bb.isInBbox(maxLon, minLat) || bb.isInBbox(maxLon, maxLat)) ) {
																	for(Map.Entry<String, PrintWriter> entry_int : storageFileMap.entrySet()) {
																		if(entry_int.getKey().equals(locationString)) {

																			for(Map.Entry<String, Integer> entry_int_int : storageFileCount.entrySet()) {
																				if(entry_int_int.getKey().equals(locationString)) {

																					boolean storeTweet = false;

																					// Management of storeMaxOneTweetEveryXseconds
																					if(storeMaxOneTweetEveryXseconds != null && storeMaxOneTweetEveryXseconds > 0l) {
																						Long lastTimestamp = storageFileLastTimestamp.get(locationString);
																						if(lastTimestamp != null && (System.currentTimeMillis() - lastTimestamp) < (storeMaxOneTweetEveryXseconds * 1000l)) {
																							System.out.println("SKIPPED TWEET FOR LOCATION: " + locationString + " - only  " + (System.currentTimeMillis() - lastTimestamp) + " ms (< " + storeMaxOneTweetEveryXseconds + "s)"
																									+ "since last tweet received - queue free places: " + queue.remainingCapacity());
																							storeTweet = false;
																						}
																						else {
																							storageFileLastTimestamp.put(locationString, System.currentTimeMillis());
																							storeTweet = true;
																						}
																					}
																					else {
																						// No storeMaxOneTweetEveryXseconds set, store tweet anyway
																						storeTweet = true;
																					}

																					if(storeTweet) {
																						// Store to list
																						storageFileTweetList.get(entry_int.getKey()).add(msg);
																						// Store to file
																						// entry_int.getValue().write(msg);
																						// entry_int.getValue().flush();

																						inOneBbox = true;
																						totalTweetStoredCount++;

																						// Increment number of stored tweets for the bounding box
																						Integer storageCount = entry_int_int.getValue();
																						storageFileCount.put(locationString, storageCount + 1);
																						System.out.println("RECEIVED (lat, lng) GEOLOCATED TWEET: " + locationString + " tot: " + (storageCount + 1) + " - queue free places: " + queue.remainingCapacity());
																					}

																				}
																			}

																			if(totalTweetStoredCount % 100 == 0) {
																				printMemoryStatus();
																				System.gc();
																				System.out.println("GARBAGE COLLECTOR CALLED: ");
																				printMemoryStatus();
																			}

																		}
																	}							
																}
															}

															if(!inOneBbox) {
																logFileList.add("DISCARTED INTERSECTING PLACE: " + ((receivedStatus.getPlace().getName() != null) ? receivedStatus.getPlace().getName() : "NULL") +
																		"\n WITH AREA: " + areaInKmQuad + " Km2 (" + latitudeDiff + " * " + longitudeDiff + ") > NOT INTERSECTING BBOX.\n");
															}

														}
														else {
															logFileList.add("IGNORED INTERSECTING PLACE: " + ((receivedStatus.getPlace().getName() != null) ? receivedStatus.getPlace().getName() : "NULL") +
																	" WITH AREA: " + areaInKmQuad + " Km2 (" + latitudeDiff + " * " + longitudeDiff + ") > " + maxAreaInKmQuad + " km2 max area.\n");
														}
													}
												}
											}
											else {
												logFileList.add("PLACE WITH COORDINATE BBOX FIRST LINE OF LENGTH != 4 (" + plCoord.length + ") > " +
														"name: " + ((receivedStatus.getPlace().getName() != null) ? receivedStatus.getPlace().getName() : "NULL") +
														" - id: " + ((receivedStatus.getPlace().getId() != null) ? receivedStatus.getPlace().getId() : "NULL") +
														"\n");
											}

										}
										else {
											logFileList.add("PLACE WITH COORDINATE BBOX EMPTY > " +
													"name: " + ((receivedStatus.getPlace().getName() != null) ? receivedStatus.getPlace().getName() : "NULL") +
													" - id: " + ((receivedStatus.getPlace().getId() != null) ? receivedStatus.getPlace().getId() : "NULL") +
													"\n");
										}

									}
									else {
										logFileList.add("PLACE WITHOUT COORDINATES > " +
												"name: " + ((receivedStatus.getPlace().getName() != null) ? receivedStatus.getPlace().getName() : "NULL") +
												" - id: " + ((receivedStatus.getPlace().getId() != null) ? receivedStatus.getPlace().getId() : "NULL") +
												"\n");
									}
								}
								else {
									logFileList.add("TWEET NOT GEOLOCATED\n");
								}
							}
						} catch (Exception e) {
							logFileList.add("Exception " + ((e != null && e.getMessage() != null) ? e.getMessage() : "---") + "\n");
							System.out.println("Exception " + ((e != null && e.getMessage() != null) ? e.getMessage() : "---") + "\n");
							e.printStackTrace();
						}

					}
					else {
						logFileList.add(sdf.format(new Date()) + " - ERROR CODE: " + msg + "\n");
						logger.info(sdf.format(new Date()) + " - ERROR CODE: " + msg);
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

		TwitterSTREAMBboxCrawler crawler = new TwitterSTREAMBboxCrawler();

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
			String bboxesListFilePath = propManager.getProperty(PropertyManager.STREAMbboxListPath);
			File bboxesListFile = new File(bboxesListFilePath);
			if(bboxesListFile == null || !bboxesListFile.exists() || !bboxesListFile.isFile()) {
				System.out.println("ERROR: bounding box list input file path (property '" + PropertyManager.STREAMbboxListPath + "')"
						+ " wrongly specified > PATH: '" + ((bboxesListFilePath != null) ? bboxesListFilePath : "NULL") + "'");
				if(bboxesListFile != null && !bboxesListFile.exists()) {
					System.out.println("      The file does not exist!"); 
				}
				if(bboxesListFile != null && bboxesListFile.exists() && !bboxesListFile.isFile()) {
					System.out.println("      The path does not point to a valid file!"); 
				}
			}
			else {
				crawler.fullPathOfBoundingBoxesFile = bboxesListFilePath;
			}
		} catch (Exception e) {
			System.out.println("ERROR: bounding box list input file path (property '" + PropertyManager.STREAMbboxListPath + "')"
					+ " wrongly specified - exception: " + ((e.getMessage() != null) ? e.getMessage() : "NULL"));
		}


		// Load full path of output directory
		try {
			String outputDirectoryFilePath = propManager.getProperty(PropertyManager.STREAMbboxFullPathOfOutputDir);
			File outputDirFile = new File(outputDirectoryFilePath);
			if(outputDirFile == null || !outputDirFile.exists() || !outputDirFile.isDirectory()) {
				System.out.println("ERROR: output directory full path (property '" + PropertyManager.STREAMbboxFullPathOfOutputDir + "')"
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
			System.out.println("ERROR: output directory full path (property '" + PropertyManager.STREAMbboxFullPathOfOutputDir + "')"
					+ " wrongly specified - exception: " + ((e.getMessage() != null) ? e.getMessage() : "NULL"));
			return;
		}

		// Output format
		try {
			String outputFormat = propManager.getProperty(PropertyManager.STREAMbboxOutputFormat);

			if(outputFormat != null && outputFormat.trim().toLowerCase().equals("json")) {
				crawler.outpuTweetFormat = "json";
			}
			else if(outputFormat != null && outputFormat.trim().toLowerCase().equals("tab")) {
				crawler.outpuTweetFormat = "tab";
			}
			else {
				crawler.outpuTweetFormat = "json";
				System.out.println("Impossible to read the '" + PropertyManager.STREAMbboxOutputFormat + "' property - set to: " + crawler.outpuTweetFormat);
			}

		} catch (Exception e) {
			System.out.println("ERROR: output format (property '" + PropertyManager.STREAMbboxOutputFormat + "') - exception: " + ((e.getMessage() != null) ? e.getMessage() : "NULL"));
			return;
		}

		// Limit by one tweet per X seconds
		try {
			String limitRate = propManager.getProperty(PropertyManager.STREAMbboxLimitByOneTweetPerXsec);

			if(limitRate != null && !limitRate.trim().equals("")) {
				try {
					crawler.storeMaxOneTweetEveryXseconds = Long.valueOf(limitRate);
				}
				catch(Exception e) {
					crawler.storeMaxOneTweetEveryXseconds = -1l;
					System.out.println("Impossible to read the '" + PropertyManager.STREAMbboxLimitByOneTweetPerXsec + "' property - set to: " + crawler.storeMaxOneTweetEveryXseconds);
				}
			}
			else {
				crawler.storeMaxOneTweetEveryXseconds = -1l;
			}

		} catch (Exception e) {
			System.out.println("ERROR: output format (property '" + PropertyManager.STREAMbboxLimitByOneTweetPerXsec + "') - exception: " + ((e.getMessage() != null) ? e.getMessage() : "NULL"));
			return;
		}

		// Language filter
		try {
			String langFilter = propManager.getProperty(PropertyManager.STREAMbboxLimitByLanguage);

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
				System.out.println("Impossible to read the '" + PropertyManager.STREAMbboxLimitByLanguage + "' property - Language filter not set");
			}

		} catch (Exception e) {
			System.out.println("ERROR: output format (property '" + PropertyManager.STREAMbboxLimitByLanguage + "') - exception: " + ((e.getMessage() != null) ? e.getMessage() : "NULL"));
			return;
		}

		// Flush tweet to files every X tweet received
		try {
			String flushRate = propManager.getProperty(PropertyManager.STREAMbboxFlushToFileEveryXtweetsCrawled);

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
					System.out.println("Impossible to read the '" + PropertyManager.STREAMbboxFlushToFileEveryXtweetsCrawled + "' property - set to: " + crawler.flushFileNumTweets);
				}
			}
			else {
				crawler.flushFileNumTweets = 100;
			}

		} catch (Exception e) {
			System.out.println("ERROR: output format (property '" + PropertyManager.STREAMbboxFlushToFileEveryXtweetsCrawled + "') - exception: " + ((e.getMessage() != null) ? e.getMessage() : "NULL"));
			return;
		}

		// Change storage file every X tweet received
		try {
			String changeFileRate = propManager.getProperty(PropertyManager.STREAMbboxChangeStorageFileEveryXtweetsCrawled);

			if(changeFileRate != null && !changeFileRate.trim().equals("")) {
				try {
					crawler.changeFileNumTweets = Integer.valueOf(changeFileRate);

					if(crawler.changeFileNumTweets < 200) {
						crawler.changeFileNumTweets = 200;
					}
				}
				catch(Exception e) {
					crawler.changeFileNumTweets = 20000;
					System.out.println("Impossible to read the '" + PropertyManager.STREAMbboxChangeStorageFileEveryXtweetsCrawled + "' property - set to: " + crawler.changeFileNumTweets);
				}
			}
			else {
				crawler.changeFileNumTweets = 20000;
			}

		} catch (Exception e) {
			System.out.println("ERROR: output format (property '" + PropertyManager.STREAMbboxChangeStorageFileEveryXtweetsCrawled + "') - exception: " + ((e.getMessage() != null) ? e.getMessage() : "NULL"));
			return;
		}


		// Loading bounding boxes from file
		try {
			BufferedReader in = new BufferedReader(new InputStreamReader(new FileInputStream(new File(crawler.fullPathOfBoundingBoxesFile)), "UTF-8"));

			String str;
			while ((str = in.readLine()) != null) {
				if(!str.trim().equals("")) {
					// Add bbox to crawl
					String[] bboxNameCoordSplit = str.split("\t");

					if(bboxNameCoordSplit != null && bboxNameCoordSplit.length == 2 && 
							bboxNameCoordSplit[0] != null && bboxNameCoordSplit[0].trim().length() > 0 &&
							bboxNameCoordSplit[1] != null && bboxNameCoordSplit[1].trim().length() > 0) {

						String[] bboxCoordinatesSplit = bboxNameCoordSplit[1].trim().split(",");

						if(bboxCoordinatesSplit != null && bboxCoordinatesSplit.length == 4) {
							Double lngSW = null;
							Double latSW = null;
							Double lngNE = null;
							Double latNE = null;

							try { lngSW = Double.valueOf(bboxCoordinatesSplit[0]);	} catch(Exception e) { /* Do nothintg */}
							try { latSW = Double.valueOf(bboxCoordinatesSplit[1]);	} catch(Exception e) { /* Do nothintg */}
							try { lngNE = Double.valueOf(bboxCoordinatesSplit[2]);	} catch(Exception e) { /* Do nothintg */}
							try { latNE = Double.valueOf(bboxCoordinatesSplit[3]);	} catch(Exception e) { /* Do nothintg */}

							if(lngSW != null && latSW != null && lngNE != null && latNE != null) {
								Bbox boundingBoxObject = new Bbox(lngSW, latSW, lngNE, latNE);

								crawler.trackBbox.put(bboxNameCoordSplit[0], boundingBoxObject);
							}
							else {
								System.out.println("Impossible to parse bounding box coordinates values of the following line of the bounding box file '" + ((str != null) ? str : "NULL") + "'");
							}

						}
						else {
							System.out.println("Impossible to parse bounding box coordinates of the following line of the bounding box file '" + ((str != null) ? str : "NULL") + "'");
						}



					}
					else {
						System.out.println("Impossible to read the following line of the bounding box file '" + ((str != null) ? str : "NULL") + "'");
					}
				}
			}

			in.close();
		}
		catch (Exception e) {
			System.out.println("Exception reading bounding boxes names and coordinates from file: " +  e.getMessage() + " > PATH: '" + ((crawler.fullPathOfBoundingBoxesFile != null) ? crawler.fullPathOfBoundingBoxesFile : "NULL") + "'");
		}

		crawler.outputDir = new File(crawler.outputDirPath);

		// Printing arguments:
		System.out.println("\n***************************************************************************************");
		System.out.println("******************** LOADED PARAMETERS ************************************************");
		System.out.println("   > Property file loaded from path: '" + ((args[0].trim() != null) ? args[0].trim() : "NULL") + "'");
		System.out.println("        PROPERTIES:");
		System.out.println("           - NUMBER OF TWITTER API CREDENTIALS: " + ((crawler.consumerKey != null) ? crawler.consumerKey.size() : "ERROR"));
		System.out.println("           - LANGUAGE FILTER: " + ((crawler.langList != null) ? crawler.langList : "ERROR"));
		System.out.println("           - PATH OF LIST OF BOUNDING BOXES TO CRAWL: '" + ((crawler.fullPathOfBoundingBoxesFile != null) ? crawler.fullPathOfBoundingBoxesFile : "NULL") + "'");
		System.out.println("           - PATH OF CRAWLER OUTPUT FOLDER: '" + ((crawler.outputDirPath != null) ? crawler.outputDirPath : "NULL") + "'");
		System.out.println("           - OUTPUT FORMAT: '" + ((crawler.outputDirPath != null) ? crawler.outputDirPath : "NULL") + "'");
		System.out.println("           - STORE TWEETS TO FILE EVERY " + ((crawler.flushFileNumTweets != null) ? crawler.flushFileNumTweets : "NULL") + " TWEETS CRAWLED (MIN. ALLOWED VAL 20, DEFAULT VALUE 100)");
		System.out.println("           - SWITCH TO NEW TWEETS STORAGE FILE EVERY " + ((crawler.changeFileNumTweets != null) ? crawler.changeFileNumTweets : "NULL") + " TWEETS CRAWLED (MIN. ALLOWED VAL 200, DEFAULT VALUE 20000)");
		System.out.println("               (STORAGE FILES AND COUNTERS OF CRAWLED TWEETS ARE MANAGED SEPARATELY, ONE FOR EACH BOUNDING BOX)");
		System.out.println("   -");
		System.out.println("   NUMBER OF BOUNDING BOXES / LINES READ FROM THE LIST: " + ((crawler.trackBbox != null) ? crawler.trackBbox.size() : "READING ERROR"));
		System.out.println("***************************************************************************************\n");

		if(crawler.trackBbox == null || crawler.trackBbox.size() == 0) {
			System.out.println("Empty list of bounding boxes to crawl > EXIT");
			return;
		}

		if(crawler.consumerKey == null || crawler.consumerKey.size() == 0) {
			System.out.println("Empty list of valid Twitter API credentials > EXIT");
			return;
		}

		System.out.println("<><><><><><><><><><><><><><><><><><><>");
		System.out.println("List of bounding boxes to crawl:");
		int bboxCounter = 1;
		for(Entry<String, Bbox> bboxElem : crawler.trackBbox.entrySet()) {
			System.out.println(bboxCounter++ + " Bbox: " + bboxElem.getKey() + " > Coords: " + bboxElem.getValue().toString() + ", AREA: " + bboxElem.getValue().getArea());
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
