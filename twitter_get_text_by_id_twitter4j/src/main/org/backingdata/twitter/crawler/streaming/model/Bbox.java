package main.org.backingdata.twitter.crawler.streaming.model;

public class Bbox {
	// BBOX format: (lngSW, latSW, lngNE, latNE)
	private double lngSW = 0d;
	private double latSW = 0d;
	private double lngNE = 0d;
	private double latNE = 0d;
	
	public static final double constAreaKm = 0.0089982311916d;
	
	// Constructors
	public Bbox() {
		super();
		this.lngSW = 0d;
		this.latSW = 0d;
		this.lngNE = 0d;
		this.latNE = 0d;
	}
	
	public Bbox(double lngSW, double latSW, double lngNE, double latNE) {
		super();
		this.lngSW = (-180d <= lngSW && 180d >= lngSW)? lngSW : 0d;
		this.latSW = (-90d <= latSW && 90d >= latSW)? latSW : 0d;
		this.lngNE = (-180d <= lngNE && 180d >= lngNE)? lngNE : 0d;
		this.latNE = (-90d <= latNE && 90d >= latNE)? latNE : 0d;
	}
	
	
	// Getters and setters
	public double getLngSW() {
		return lngSW;
	}
	public void setLngSW(double lngSW) {
		this.lngSW = (-180d <= lngSW && 180d >= lngSW)? lngSW : 0d;
	}
	public double getLatSW() {
		return latSW;
	}
	public void setLatSW(double latSW) {
		this.latSW = (-90d <= latSW && 90d >= latSW)? latSW : 0d;
	}
	public double getLngNE() {
		return lngNE;
	}
	public void setLngNE(double lngNE) {
		this.lngNE = (-180d <= lngNE && 180d >= lngNE)? lngNE : 0d;
	}
	public double getLatNE() {
		return latNE;
	}
	public void setLatNE(double latNE) {
		this.latNE = (-90d <= latNE && 90d >= latNE)? latNE : 0d;
	}
	
	
	// Utility methods
	public double[][] getCoordinateArray() {
		double[][] bb = {{this.getLngSW(), this.getLatSW()}, {this.getLngNE(), this.getLatNE()}};
		return bb;
	}
	
	public boolean isInBbox(double lng, double lat) {
		
		boolean lngIn = false;
		if(this.lngSW <= this.lngNE) {
			if(this.lngSW <= lng && lng <= this.lngNE) {
				lngIn = true;
			}
		}
		else {
			if( (this.lngSW >= lng && lng <= 180d) || (-180d <= lng && lng <= this.lngNE) ) {
				lngIn = true;
			}
		}
		
		boolean latIn = false;
		// Cannot select a bounding box where latSW > latNE
		if(this.latSW <= this.latNE) {
			if(this.latSW <= lat && lat <= this.latNE) {
				latIn = true;
			}
		}
		
		return lngIn && latIn;
	}
	
	public double getArea() { 
		double latDist = 0d;
		double lngDist = 0d;
		
		latDist = this.latNE - this.latSW;
		lngDist = this.lngNE - this.lngSW;
		
		return latDist * lngDist;
	}

	@Override
	public String toString() {
		return "Bbox [lngSW=" + lngSW + ", latSW=" + latSW + ", lngNE=" + lngNE + ", latNE=" + latNE + "]";
	}
	
	
}
