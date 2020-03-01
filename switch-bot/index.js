const net = require("net");

const socket = net.connect(6000, "192.168.1.11", () => {
  console.log("Connected to Switch!");
  socket.write('configure echoCommands 1\r\n', 'utf8');

});

socket.on("data", (data) => {
  console.log(`Command Read: ${data.toString()}`);
});

socket.on('close', () => {
  console.log('Disconnected from switch!');
});



/*
for (let j = 0; j < 9; j++)
{
  console.log(`Month: ${j}`);
  for (let i = 0; i < 28; i++)
  {
    console.log(`Day: ${i}`);
    socket.write('click A\r\n');

    socket.write('click DLEFT\r\n');
    socket.write('click DLEFT\r\n');
    socket.write('click DLEFT\r\n');
    socket.write('click DLEFT\r\n');
    socket.write('click DLEFT\r\n');

    socket.write('click DUP\r\n');

    socket.write('click DRIGHT\r\n');
    socket.write('click DRIGHT\r\n');
    socket.write('click DRIGHT\r\n');
    socket.write('click DRIGHT\r\n');
    socket.write('click DRIGHT\r\n');

    socket.write('click A\r\n');
  }
  socket.write('click A\r\n');

  socket.write('click DLEFT\r\n');
  socket.write('click DLEFT\r\n');
  socket.write('click DLEFT\r\n');
  socket.write('click DLEFT\r\n');
  socket.write('click DLEFT\r\n');
  socket.write('click DLEFT\r\n');

  socket.write('click DUP\r\n');

  socket.write('click DRIGHT\r\n');
  socket.write('click DRIGHT\r\n');
  socket.write('click DRIGHT\r\n');
  socket.write('click DRIGHT\r\n');
  socket.write('click DRIGHT\r\n');
  socket.write('click DRIGHT\r\n');

  socket.write('click A\r\n');
}
*/