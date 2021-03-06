require('dotenv').config(); //initialize dotenv
const { Client, Intents } = require('discord.js');

const client = new Client({ intents: [Intents.FLAGS.GUILDS, Intents.FLAGS.GUILD_MESSAGES] });

client.on('ready', () => {
  console.log(`Logged in as ${client.user.tag}!`);
});

var prefix = "$";

client.login(process.env.CLIENT_TOKEN); //login bot using token
