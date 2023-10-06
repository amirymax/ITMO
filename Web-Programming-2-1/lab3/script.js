let rows = 0;
const tablee = document.createElement('table');
function create(){
    if (document.getElementById('table')){
        alert("Table already created");
    }else{
        tablee.innerHTML = "<table>";
        tablee.setAttribute('id', 'table');
        document.body.append(tablee);
        rows += 1;
        add_row();}
}
function add_row(){
    let row = tablee.insertRow();
    row.insertCell().append(rows);
    row.insertCell().append("something unneeded");
    row.insertCell().append("something beautiful");
    rows += 1;
}
function remove_row(){
    if (document.getElementById('number').value === "") {
        alert("Write the number of row which you wanna delete")
    }
    row_number = document.getElementById('number').value;
    try {
        tablee.deleteRow(row_number - 1);
    }catch (e) {
        alert("The number of row you entered is incorrect");
    }
}
function toallow(){
    var add_btn = document.getElementById('add');
    var delete_btn = document.getElementById('remove');
    var num_place = document.getElementById('number');

    add_btn.disabled = !add_btn.disabled;
    delete_btn.disabled = !delete_btn.disabled;
    num_place.disabled = !num_place.disabled;
}

