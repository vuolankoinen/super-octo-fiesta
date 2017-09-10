import org.jsoup.Jsoup;
import org.jsoup.select.Elements;
import org.jsoup.nodes.Element;
import org.jsoup.nodes.Document;

import java.io.IOException;

import javax.json.Json;
import javax.json.JsonObject;
import javax.json.JsonReader;

import java.io.StringReader;  // Saa stringin streamiksi

class ahmatti {
    public static void main(String[] args) {
	System.out.println("Hei! Ladataan sivua, odota hetki. :)");
	for (int tapaus = 8; tapaus < 12; ++tapaus) {
	    try {
		Document valtuusto = Jsoup.connect("https://dev.hel.fi/paatokset/v1/issue/"+Integer.toString(tapaus)+"/?format=json").ignoreContentType(true).get();
		StringReader runko = new StringReader(valtuusto.body().html());
		JsonReader jsonLukija = Json.createReader(runko);
		JsonObject jsonOlio = jsonLukija.readObject();
		System.out.println( jsonOlio.getString("subject"));
	    } catch (IOException e) {
		System.out.println("Virhe! Sivua ei luettu: " + e.getMessage());
	    }
	}
    }
}
