function process_text(text)
{

out = text.toLowerCase().trim().replace(/[^a-z]/g,'');
out_arr = out.split(" ");
return out_arr;
}


async function loadDict() 
{
await $.ajax({url: 'dict.csv',datatype: 'text',}).done(success);

}


