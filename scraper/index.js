const siteUrl = "https://www.serebii.net/swordshield/maxraidbattles/den#.shtml";
const axios = require("axios");
const cheerio = require("cheerio");
const fs = require("fs");


const fetchData = async (denNum) => {
  const siteResult = await axios.get(siteUrl.replace(/#/, denNum));
  return cheerio.load(siteResult.data);
};

const getResults = async () => {
  let denArr = [];
  for (i = 1; i < 94; i++) {
    const $ = await fetchData(i);

    let pkmnArr = [];
    let count = 0;

    $('td', '.trainer').each((index, element) => {
      let pkmnObj = new Object;
      if ($(element).text().length <= 1)
        return;

      else if ($(element).attr('class') == "pkmn")
      {
        pkmnObj.name = $(element).text();
        pkmnArr.push(pkmnObj);
      }

      else if ($(element).text().startsWith("Abilitys?"))
      {
        pkmnArr[count++].ability = $(element).text().replace(/abilitys\?/i, "");
      }

    })
    console.log(pkmnArr);
    let denObj = {
      den: i.toString(),
      pkmn: pkmnArr
    }
    denArr.push(denObj);
  }

  console.log(denArr);
  let dens = JSON.stringify(denArr);
  fs.writeFileSync("scraper_results.json", dens);
}

getResults();