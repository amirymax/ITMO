<?php
$products = array(
    array(
        'id' => 1,
        'name' => 'Steve Jobs',
        'group' => 52211,
        'isu' => 943053
    ),
    array(
        'id' => 2,
        'name' => 'Bill Gates',
        'group' => 4423,
        'isu' => 737009
    ),
    array(
        'id' => 3,
        'name' => 'Mark Zuckerberg',
        'group' => 98221,
        'isu' => 903001
    ),
);
$xml = new DOMDocument('1.0', 'UTF-8');

$root = $xml->createElement('students');
$xml->appendChild($root);

foreach ($products as $value) {
    $product = $xml->createElement('student');
    foreach ($value as $key => $value) {
        $node = $xml->createElement($key, $value);
        $product->appendChild($node);
    }
    $root->appendChild($product);

}
$xml->formatOutput = true;
$xml->save('data.xml') or die('Error');
